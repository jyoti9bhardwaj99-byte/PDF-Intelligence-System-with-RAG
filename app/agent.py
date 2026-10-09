from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from app.config import LLM_MODEL, TOP_K

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's uploaded documents.
Always use the 'retrieve_context' tool before answering.
Answer only from the retrieved text. If the answer is not there, say you could not find it in the documents."""


def build_agent(vector_store):
    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve passages relevant to a query from the uploaded documents."""
        results = vector_store.similarity_search(query=query, k=TOP_K)
        return "\n\n".join(doc.page_content for doc in results)

    return create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def ask(agent, question, thread_id):
    """Ask one question. The thread_id keeps each conversation's memory separate."""
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread_id}},
    )
    return response["messages"][-1].content