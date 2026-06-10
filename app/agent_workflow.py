from typing import List, Literal

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.config import GROQ_API_KEY
from app.ingest import get_retriever

# ---------------------------------------------------------------------------
# Routing data model
# ---------------------------------------------------------------------------

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )


# ---------------------------------------------------------------------------
# Heuristic router (keyword-based fallback, used when LLM is unavailable)
# ---------------------------------------------------------------------------

_VECTORSTORE_KEYWORDS = {
    "agent",
    "agents",
    "prompt",
    "prompting",
    "prompt engineering",
    "adversarial",
    "adversarial attack",
    "llm attack",
}


def route_heuristic(question: str) -> Literal["vectorstore", "wiki_search"]:
    """Return the routing decision based on keyword matching.

    Routes to 'vectorstore' when the question contains keywords related to
    agents, prompt engineering, or adversarial attacks on LLMs; otherwise
    routes to 'wiki_search'.
    """
    q_lower = question.lower()
    for keyword in _VECTORSTORE_KEYWORDS:
        if keyword in q_lower:
            return "vectorstore"
    return "wiki_search"


# ---------------------------------------------------------------------------
# LLM-based router (used when GROQ_API_KEY is available)
# ---------------------------------------------------------------------------

def _build_llm_router():
    if not GROQ_API_KEY:
        return None
    try:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        structured_llm_router = llm.with_structured_output(RouteQuery)

        system = (
            "You are an expert at routing a user question to a vectorstore or wikipedia. "
            "The vectorstore contains documents related to agents, prompt engineering, and "
            "adversarial attacks. Use the vectorstore for questions on these topics. "
            "Otherwise, use wiki-search."
        )
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        return route_prompt | structured_llm_router
    except Exception:
        return None


_llm_router = None


def _get_llm_router():
    global _llm_router
    if _llm_router is None:
        _llm_router = _build_llm_router()
    return _llm_router


# ---------------------------------------------------------------------------
# Wikipedia tool
# ---------------------------------------------------------------------------

_wiki_tool = None


def _get_wiki_tool() -> WikipediaQueryRun:
    global _wiki_tool
    if _wiki_tool is None:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        _wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return _wiki_tool


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """Represents the state of the LangGraph workflow."""

    question: str
    generation: str
    documents: List[Document]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """Retrieve documents from the vector store."""
    print("--Retrieve--")
    question = state["question"]
    documents = get_retriever().invoke(question)
    return {"documents": documents, "question": question}


def wiki_search(state: GraphState) -> dict:
    """Search Wikipedia for the question."""
    print("--Wikipedia--")
    question = state["question"]
    wiki_content = _get_wiki_tool().invoke({"query": question})
    wiki_document = Document(page_content=wiki_content)
    return {"documents": [wiki_document], "question": question}


def route_request(state: GraphState) -> Literal["vectorstore", "wiki_search"]:
    """Route the question to the appropriate data source."""
    print("--ROUTE QUESTION--")
    question = state["question"]

    router = _get_llm_router()
    if router is not None:
        try:
            source = router.invoke({"question": question})
            decision = source.datasource
        except Exception:
            decision = route_heuristic(question)
    else:
        decision = route_heuristic(question)

    if decision == "wiki_search":
        print("--- ROUTE QUESTION TO Wiki search --")
        return "wiki_search"
    else:
        print("--- ROUTE QUESTION TO RAG --")
        return "vectorstore"


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------

def build_workflow():
    """Construct and compile the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)

    workflow.add_conditional_edges(
        START,
        route_request,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)

    return workflow.compile()


# Lazily compiled app graph
_compiled_app = None


def get_compiled_app():
    """Return the compiled LangGraph application (singleton)."""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_workflow()
    return _compiled_app
