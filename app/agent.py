from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path

from app.config import LLM_MODEL, MIN_SIMILARITY, NOT_FOUND_MESSAGE, TOP_K

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's uploaded documents.
Always use the 'retrieve_context' tool before answering.
Answer only from the retrieved text. Each passage starts with a [Source: file, page N] tag; mention the page when it helps.
If the tool says NO_RELEVANT_PASSAGES_FOUND, say you could not find it in the documents."""


class DocAgent:
    """Wraps the agent and remembers which chunks the last question used."""

    def __init__(self, vector_store):
        self.sources = []       # chunks used for the latest question
        self.searched = False   # did the AI search at all for this question?
        self.agent = self._build(vector_store)

    def _build(self, vector_store):
        llm = ChatGroq(model=LLM_MODEL, temperature=0)

        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve passages relevant to a query from the uploaded documents."""
            self.searched = True
            results = vector_store.similarity_search_with_score(query=query, k=TOP_K)

            # block only when even the BEST chunk is irrelevant (e.g. the cricket question)
            best_score = max((score for _, score in results), default=0)
            if best_score < MIN_SIMILARITY:
                return "NO_RELEVANT_PASSAGES_FOUND"
            good = results

            parts = []
            for doc, score in good:
                file = Path(doc.metadata.get("source", "document")).name
                page = doc.metadata.get("page", 0) + 1  # PyPDFLoader counts pages from 0
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

        # The AI searched but nothing was similar enough: don't trust any answer
        if self.searched and not self.sources:
            return NOT_FOUND_MESSAGE, []

        # One entry per (file, page), keeping the best score
        best = {}
        for s in self.sources:
            key = (s["file"], s["page"])
            if key not in best or s["score"] > best[key]["score"]:
                best[key] = s
        return answer, sorted(best.values(), key=lambda s: s["score"], reverse=True)


def build_agent(vector_store):
    return DocAgent(vector_store)


def ask(agent, question, thread_id):
    return agent.ask(question, thread_id)