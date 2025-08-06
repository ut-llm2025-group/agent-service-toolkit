import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)
from agents.lightrag_agent.utils import RagSystem, LIGHTRAG_WORKING_DIR
from agents.naiverag_agent.utils import FaissRagSystem, FAISS_DB_DIR

from schema import ChatMessage
from core.settings import settings


logger = logging.getLogger(__name__)

def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


async def process_documents(files: list, parsing_methods: list[str], llm_model: str):
    """
    Main controller to process uploaded documents based on the selected parsing method.

    Args:
        files (List): A list of uploaded file objects from FastAPI.
        parsing_method (str): The method to use ('Chunking', 'Graph Extraction', 'Both').
    """
    logger.info(f"Starting document processing with method: {parsing_methods}")
    # Combine content from all uploaded files into a single text block
    full_text = ""
    for file in files:
        content = await file.read()
        full_text += content.decode("utf-8") + "\n\n"

    if "Chunking" in parsing_methods:
        logger.info("Initializing FAISS RAG System for Chunking...")
        faiss_system = FaissRagSystem(db_path=FAISS_DB_DIR, llm_model=llm_model)
        await faiss_system.insert_text(full_text)
        logger.info("FAISS knowledge base updated.")
        return {"message": "Knowledge base updated using FAISS (Chunking)."}

    if "Graph Extraction" in parsing_methods:
        rag_system = RagSystem(working_dir=LIGHTRAG_WORKING_DIR, llm_model=llm_model)
        try:
            await rag_system.initialize()
            await rag_system.insert_text(full_text)
            logger.info("LightRAG knowledge base updated.")
            return {"message": f"Knowledge base updated using LightRAG (Graph Extraction)."}
        except Exception as e:
            logger.error(f"An error occurred during LightRAG processing: {e}")
            raise
        finally:
            # Ensure resources are cleaned up
            await rag_system.finalize()
