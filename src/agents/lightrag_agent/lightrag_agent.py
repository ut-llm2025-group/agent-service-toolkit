from typing import TypedDict
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    message: str

def node1(state: GraphState) -> GraphState:
    """A node that appends a string to the message in the state."""
    print("---Executing Node 1---")
    current_message = state["message"]
    new_message = current_message + " I reached Node 1."
    return {"message": new_message}

workflow = StateGraph(GraphState)

workflow.add_node("node_1", node1)

workflow.set_entry_point("node_1")
workflow.add_edge("node_1", END)

app = workflow.compile()
