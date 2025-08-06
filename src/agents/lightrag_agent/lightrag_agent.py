import asyncio

from typing import TypedDict
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from .utils import RagSystem, WORKING_DIR, LLM_MODEL


class AgentState(MessagesState, total=False):
    ...


rag_system = RagSystem(WORKING_DIR, LLM_MODEL)


def retrieve_generation(state: AgentState) -> AgentState:
    """A node that appends a string to the message in the state."""
    asyncio.run(rag_system.initialize())
    result = asyncio.run(rag_system.query(state["messages"][-1].content, mode="hybrid", top_k=20, chunk_top_k=5))
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