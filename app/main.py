from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent_workflow import get_compiled_app
from app.database import init_sqlite_db

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart RAG Assistant",
    description="LangGraph-powered RAG agent backed by Astra DB and Wikipedia.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the SQLite telemetry database on startup."""
    init_sqlite_db()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Health-check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a question through the RAG agent and return an answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    graph = get_compiled_app()
    inputs = {"question": request.question}

    final_output = None
    for output in graph.stream(inputs):
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="Agent produced no output.")

    answer = ""
    source = ""

    for key, value in final_output.items():
        source = key  # "retrieve" or "wiki_search"

        if "generation" in value and value["generation"]:
            answer = value["generation"]
        elif "documents" in value and value["documents"]:
            doc = value["documents"][0]
            # Prefer metadata description; fall back to raw page content
            answer = doc.metadata.get("description", "") or doc.page_content

    return QueryResponse(answer=answer, source=source)
