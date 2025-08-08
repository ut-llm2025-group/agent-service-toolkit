import logging
import asyncio

from typing import TypedDict
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .utils import RagSystem, LIGHTRAG_WORKING_DIR
from core.settings import settings


logger = logging.getLogger(__name__)
class AgentState(MessagesState, total=False):
    ...


def retrieve_generation(state: AgentState, config: RunnableConfig) -> AgentState:
    """A node that appends a string to the message in the state."""
    model = config["configurable"].get("model", settings.DEFAULT_MODEL)
    thread_id = config["configurable"].get("thread_id", None)
    if model == "ollama":
        model = settings.OLLAMA_MODEL
    else:
        model = "gpt-4o-mini"
    logger.info(f"Using model: {model} for RAG system.")
    rag_system = RagSystem(LIGHTRAG_WORKING_DIR + "-" + model, model)
    asyncio.run(rag_system.initialize())
    result = asyncio.run(
        rag_system.query(state["messages"][-1].content, mode="hybrid", top_k=20, chunk_top_k=5, ids=[thread_id])
    )
    messages = state["messages"]
    ai_message = AIMessage(result)
    return {"messages": messages + [ai_message]}

workflow = StateGraph(AgentState)

workflow.add_node("retrieve_generation", retrieve_generation)

workflow.set_entry_point("retrieve_generation")
workflow.add_edge("retrieve_generation", END)

lightrag_agent = workflow.compile()


if __name__ == "__main__":
    message = HumanMessage("What is the main theme of the story?")
    result = lightrag_agent.invoke(input=MessagesState(messages=[message]))
    print(result)