import asyncio

from typing import TypedDict
from langgraph.graph import StateGraph, END
from .utils import RagSystem, WORKING_DIR


rag_system = RagSystem(WORKING_DIR)

class GraphState(TypedDict):
    message: str

def retrieve_generation(state: GraphState) -> GraphState:
    """A node that appends a string to the message in the state."""
    asyncio.run(rag_system.initialize())
    question = "What is the main theme of the story?"
    result = asyncio.run(rag_system.query(question, mode="hybrid", top_k=20, chunk_top_k=5))
    return {"message": result}

workflow = StateGraph(GraphState)

workflow.add_node("node_1", retrieve_generation)

workflow.set_entry_point("node_1")
workflow.add_edge("node_1", END)

lightrag_agent = workflow.compile()
