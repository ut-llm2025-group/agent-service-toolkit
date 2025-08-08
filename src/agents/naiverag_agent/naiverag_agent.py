import asyncio
from typing import TypedDict

from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from core.settings import settings
from .utils import FaissRagSystem, FAISS_DB_DIR 

class AgentState(MessagesState, total=False):
    """
    Represents the state of our RAG agent. It holds the history of messages.
    """
    ...

async def run_naive_rag(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    This node initializes the FaissRagSystem, retrieves the user's last question,
    queries the RAG system, and appends the AI's answer to the message list.
    """
    model = config["configurable"].get("model", settings.DEFAULT_MODEL)
    thread_id = config["configurable"].get("thread_id", None)
    if model == "ollama":
        model = settings.OLLAMA_MODEL
    else:
        model = "gpt-4o-mini"
    
    rag_system = FaissRagSystem(db_path=FAISS_DB_DIR + "-" + model, llm_model=model)
    
    last_message = state["messages"][-1]
    question = last_message.content
    
    result_text = await rag_system.query(question, doc_id=thread_id)
    
    ai_message = AIMessage(content=result_text)
    
    return {"messages": state["messages"] + [ai_message]}


naive_rag_workflow = StateGraph(AgentState)

naive_rag_workflow.add_node("naive_rag_node", run_naive_rag)

naive_rag_workflow.set_entry_point("naive_rag_node")

naive_rag_workflow.add_edge("naive_rag_node", END)

naiverag_agent = naive_rag_workflow.compile()


if __name__ == "__main__":
    
    initial_message = HumanMessage(content="What is the main theme of the story?")
    
    initial_state = MessagesState(messages=[initial_message])

    async def main():
        config = {"configurable": {"model": "ollama"}} 
        
        result = await naiverag_agent.ainvoke(initial_state, config=config)
        
        print("--- Conversation History ---")
        for msg in result['messages']:
            print(f"[{msg.type.upper()}]: {msg.content}")

        print("\n--- Final Answer ---")
        final_answer = result['messages'][-1].content
        print(final_answer)

    asyncio.run(main())