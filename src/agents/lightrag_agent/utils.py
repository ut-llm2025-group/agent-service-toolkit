import asyncio
import os
from typing import Literal, Optional
import logging

from lightrag import LightRAG, QueryParam  # type: ignore
from lightrag.kg.shared_storage import \
    initialize_pipeline_status  # type: ignore
from lightrag.llm.ollama import ollama_embed  # type: ignore
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc  # type: ignore
from lightrag.utils import setup_logger

logger = logging.getLogger("lightrag")
logger.setLevel(logging.INFO)

WORKING_DIR = "./rag_storage_test"
class RagSystem:
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.mkdir(self.working_dir)
        self.rag: Optional[LightRAG] = None
    
    async def initialize(self):
        """Initialize the RAG system asynchronously."""
        logger.info("Initializing LightRAG system...")
        self.rag = await self.initialize_rag()

    async def initialize_rag(self):
        try:
            rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_name="llama3.1:latest",
                summary_max_tokens=8192,
                llm_model_kwargs={
                    "host": "http://localhost:11434",
                    "options": {"num_ctx": 8192},
                    "timeout": 300,
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts,
                        embed_model="nomic-embed-text",
                        host="http://localhost:11434",
                    ),
                ),
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


# Example usage
async def main():
    rag_system = RagSystem(WORKING_DIR)
    try:
        # Initialize the RAG system
        await rag_system.initialize()
        
        # Example text to insert
        # with open("share/story.txt") as f:
        #     text = f.read()
        # await rag_system.insert_text(text)

        # Example query
        question = "What is the main theme of the story?"
        result = await rag_system.query(question, mode="hybrid", top_k=20, chunk_top_k=5)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await rag_system.finalize()
        logger.info("RAG system operations completed.")

    print("\n\n", result)

if __name__ == "__main__":
    setup_logger("lightrag", level="INFO")
    asyncio.run(main())