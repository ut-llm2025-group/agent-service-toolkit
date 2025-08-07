import asyncio
import os
from typing import Literal, Optional
import logging

from lightrag import LightRAG, QueryParam  # type: ignore
from lightrag.kg.shared_storage import \
    initialize_pipeline_status  # type: ignore
from lightrag.llm.ollama import ollama_embed  # type: ignore
from lightrag.llm.ollama import ollama_model_complete
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc  # type: ignore
from lightrag.utils import setup_logger

from core.settings import settings

logger = logging.getLogger("lightrag")
logger.setLevel(logging.INFO)

LIGHTRAG_WORKING_DIR = "/app/agents/lightrag_agent/rag_storage_test"
class RagSystem:
    def __init__(self, working_dir: str, llm_model: str):
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.mkdir(self.working_dir)
        self.rag: Optional[LightRAG] = None
        self.llm_model = llm_model 
    
    async def initialize(self):
        """Initialize the RAG system asynchronously."""
        logger.info("Initializing LightRAG system...")
        self.rag = await self.initialize_rag()

    async def initialize_rag(self) -> LightRAG:  
        try:
            if self.llm_model == "llama3.1:latest":
                rag = LightRAG(
                    working_dir=self.working_dir,
                    llm_model_func=ollama_model_complete,
                    llm_model_name=settings.OLLAMA_MODEL,
                    summary_max_tokens=8192,
                    llm_model_kwargs={
                        "host": settings.OLLAMA_BASE_URL,
                        "options": {"num_ctx": 8192},
                        "timeout": 300,
                    },
                    embedding_func=EmbeddingFunc(
                        embedding_dim=1536,
                        max_token_size=8192,
                        func=lambda texts: ollama_embed(
                            texts,
                            embed_model="nomic-embed-text",
                            host=settings.OLLAMA_BASE_URL,
                        ),
                    ),
                )
            elif "gpt" in self.llm_model:
                if settings.OPENAI_API_KEY is None:
                    logger.error(
                        "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
                    )
                    return 
                rag = LightRAG(
                    working_dir=self.working_dir,
                    embedding_func=EmbeddingFunc(
                        embedding_dim=1536,
                        max_token_size=8192,
                        func=lambda texts: openai_embed(
                            texts,
                        ),
                    ),
                    llm_model_func=gpt_4o_mini_complete,
                )
            # IMPORTANT: Both initialization calls are required!
            await rag.initialize_storages()  # Initialize storage backends
            await initialize_pipeline_status()  # Initialize processing pipeline
            return rag
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    async def insert_text(self, text):
        if self.rag is None:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        logger.info("Inserting text into RAG system...")
        try:
            await self.rag.ainsert(text)
            logger.info("Text inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting text: {e}")
            raise
    
    async def query(self, question: str, mode: Literal["local", "global", "hybrid", "naive"]="hybrid",
                    top_k=20, chunk_top_k=5):
        """Perform a query on the RAG system.
        Args:
            question (str): The question to query.
            mode (str): The mode of the query, default is "hybrid".
        """
        if self.rag is None:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        logger.info(f"Querying RAG system with question: {question}")
        try:
            result = await self.rag.aquery(
                question,
                param=QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, enable_rerank=False)
            )
            logger.info("Query completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise

    async def finalize(self):
        """Finalize the RAG system, cleaning up resources."""
        logger.info("Finalizing RAG system...")
        try:
            if self.rag:
                await self.rag.finalize_storages()
                logger.info("RAG system finalized successfully.")
        except Exception as e:
            logger.error(f"Error finalizing RAG system: {e}")
            raise

