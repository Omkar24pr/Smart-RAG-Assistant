import os

ASTRA_DB_ID = os.environ.get("ASTRA_DB_ID", "")
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE", None)
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HOST = os.environ.get("HOST", "0.0.0.0")
