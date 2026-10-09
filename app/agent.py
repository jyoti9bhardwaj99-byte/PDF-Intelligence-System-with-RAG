from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from app.config import LLM_MODEL, MIN_SIMILARITY, NOT_FOUND_MESSAGE, TOP_K
from app.retrieval import HybridRetriever

SYSTEM_PROMPT = """You are a careful assistant that answers questions about the user's uploaded documents.
Always use the 'retrieve_context' tool before answering.

Rules:
1. Use ONLY facts that are written in the retrieved passages. Do not add examples, explanations, or details from your own knowledge, even if they seem obvious or helpful.
2. Copy names, titles, topics, and numbers exactly as the passages write them. Do not reword project names or interview topics.
3. If a part the user asks about (for example the mini project) is not in the retrieved passages, write 'not mentioned in the retrieved text' for that part.
4. Passages start with a [Source: file, page N] tag. Cite pages like (page 6).
5. If the tool says NO_RELEVANT_PASSAGES_FOUND, say you could not find it in the documents."""


class DocAgent:
    """Wraps the agent and remembers which chunks the last question used."""

    def __init__(self, vector_store, chunks):
        self.sources = []       # chunks used for the latest question
        self.searched = False   # did the AI search at all for this question?
        self.retriever = HybridRetriever(vector_store, chunks)
        self.agent = self._build()

    def _build(self):
        llm = ChatGroq(model=LLM_MODEL, temperature=0)

        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve passages relevant to a query from the uploaded documents."""
            self.searched = True
            results, best_score = self.retriever.search(query, k=TOP_K)

            # block only when even the BEST chunk is irrelevant (e.g. the cricket question)
            if best_score < MIN_SIMILARITY:
                return "NO_RELEVANT_PASSAGES_FOUND"

            parts = []
            for doc, score in results:
                file = Path(doc.metadata.get("source", "document")).name
                page = doc.metadata.get("page", 0) + 1  # the loader counts pages from 0
                self.sources.append({
                    "file": file,
                    "page": page,
                    "score": float(score),
                    "snippet": doc.page_content[:200],
                })
                parts.append(f"[Source: {file}, page {page}]\n{doc.page_content}")
            return "\n\n".join(parts)

        return create_agent(
            model=llm,
            tools=[retrieve_context],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    def ask(self, question, thread_id):
        """Return (answer, sources) for one question."""
        self.sources = []
        self.searched = False

        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}},
        )
        answer = response["messages"][-1].content

        
                # Every real answer must be backed by sources; otherwise use the fixed message
        if not self.sources:
            return NOT_FOUND_MESSAGE, []

                # One entry per chunk (not per page), keeping the best score
        best = {}
        for s in self.sources:
            key = (s["file"], s["page"], s["snippet"])
            if key not in best or s["score"] > best[key]["score"]:
                best[key] = s
        return answer, sorted(best.values(), key=lambda s: s["score"], reverse=True)


def build_agent(vector_store, chunks):
    return DocAgent(vector_store, chunks)


def ask(agent, question, thread_id):
    return agent.ask(question, thread_id)