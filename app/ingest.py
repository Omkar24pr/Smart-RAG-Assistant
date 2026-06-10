from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.database import init_astra_db

# ---------------------------------------------------------------------------
# Source URLs (from notebook_source.py)
# ---------------------------------------------------------------------------

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# ---------------------------------------------------------------------------
# Shared objects – initialised lazily so that importing this module does not
# trigger network calls or require credentials at import time.
# ---------------------------------------------------------------------------

_embeddings = None
_astra_vector_store = None
_retriever = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def get_vector_store() -> Cassandra:
    """Return (and lazily initialise) the Cassandra vector store."""
    global _astra_vector_store
    if _astra_vector_store is None:
        init_astra_db()
        _astra_vector_store = Cassandra(
            embedding=_get_embeddings(),
            table_name="qa_mini_demo",
            session=None,
            keyspace=None,
        )
    return _astra_vector_store


def get_retriever():
    """Return (and lazily initialise) the vector-store retriever."""
    global _retriever
    if _retriever is None:
        _retriever = get_vector_store().as_retriever()
    return _retriever


def ingest_documents() -> int:
    """Load URLs, split documents, and add them to the vector store.

    Returns the total number of document chunks ingested.
    """
    docs = [WebBaseLoader(url).load() for url in URLS]
    doc_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(doc_list)

    store = get_vector_store()
    store.add_documents(docs_split)
    print(f"Inserted {len(docs_split)} documents")
    return len(docs_split)
