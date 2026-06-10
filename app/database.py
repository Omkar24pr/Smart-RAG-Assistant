import sqlite3
from pathlib import Path

import cassio

from app.config import ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID

# ---------------------------------------------------------------------------
# Cassandra / Astra DB
# ---------------------------------------------------------------------------

def init_astra_db() -> None:
    """Initialise the cassio connection to Astra DB."""
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)


# ---------------------------------------------------------------------------
# SQLite – telemetry / ingestion history
# ---------------------------------------------------------------------------

TELEMETRY_DB_PATH = Path(__file__).resolve().parent.parent / "telemetry.db"


def init_sqlite_db() -> None:
    """Create the SQLite database and required tables if they don't exist."""
    conn = sqlite3.connect(TELEMETRY_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL DEFAULT (datetime('now')),
            question  TEXT    NOT NULL,
            source    TEXT    NOT NULL,
            answer    TEXT,
            latency_ms REAL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL DEFAULT (datetime('now')),
            url       TEXT    NOT NULL,
            doc_count INTEGER NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()
