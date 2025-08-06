import os 
from typing import Optional
import logging

# For Chunking (FAISS RAG)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.settings import settings

logger = logging.getLogger(__name__)

FAISS_DB_DIR = "/app/agents/naiverag_agent/faiss_storage"
class FaissRagSystem:
    """
    A RAG system using FAISS for vector storage and LangChain for processing.
    This handles the 'Chunking' parsing method.
    """
    def __init__(self, db_path: str, llm_model: str):
        self.db_path = db_path
        self.llm_model = llm_model
        if not os.path.exists(self.db_path):
            logger.info(f"Creating FAISS storage directory: {self.db_path}")
            os.makedirs(self.db_path)

        logger.info(f"Initializing FAISS RAG with LLM model: {self.llm_model}")
        if "llama" in self.llm_model.lower():
            logger.info("Using Ollama embeddings for Llama model.")
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text", 
                base_url=settings.OLLAMA_BASE_URL
            )
        elif "gpt" in self.llm_model.lower():
            logger.info("Using OpenAI embeddings for OpenAI model.")
            self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported llm_model for FaissRagSystem: {self.llm_model}. Please use 'llama' or 'openai' compatible models.")

        self.vector_store: Optional[FAISS] = self._load_vector_store()

    def _load_vector_store(self) -> Optional[FAISS]:
        """Loads an existing FAISS vector store or returns None if not found."""
        index_path = os.path.join(self.db_path, "index.faiss")
        if os.path.exists(index_path):
            try:
                logger.info(f"Loading existing FAISS vector store from {self.db_path}")
                return FAISS.load_local(
                    self.db_path, self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. This might be due to a mismatched embedding model.")
                return None
        logger.info("No existing FAISS vector store found.")
        return None

    async def insert_text(self, text: str):
        """Chunks text, creates embeddings, and upserts them into the FAISS DB."""
        logger.info("Starting text insertion for FAISS RAG system.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")

        if self.vector_store:
            logger.info("Adding new documents to existing FAISS vector store.")
            await self.vector_store.aadd_texts(chunks)
        else:
            logger.info("Creating new FAISS vector store.")
            self.vector_store = await FAISS.afrom_texts(
                texts=chunks, embedding=self.embeddings
            )

        if self.vector_store:
            self.vector_store.save_local(self.db_path)
            logger.info(f"FAISS vector store saved to {self.db_path}")

    async def query(self, question: str) -> str:
        """Queries the FAISS vector store and generates an answer using an LLM."""
        if not self.vector_store:
            return "Error: FAISS database not found or loaded. Please add documents first."
        logger.info(f"Querying FAISS RAG system with question: {question}")

        retriever = self.vector_store.as_retriever()
        prompt_template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        if "llama" in self.llm_model.lower():
             from langchain_community.llms import Ollama
             llm = Ollama(model=self.llm_model, base_url=settings.OLLAMA_BASE_URL)
        else: 
             llm = ChatOpenAI(model_name=self.llm_model, api_key=settings.OPENAI_API_KEY)


        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return await rag_chain.ainvoke(question)