import os
import sys
import unittest
import sqlite3
from pathlib import Path

# Insert current directory into python path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

class TestInfoFinderApp(unittest.TestCase):
    
    def setUp(self):
        # Configure variables for clean local testing
        os.environ["ASTRA_DB_APPLICATION_TOKEN"] = "AstraCS:foEPIPxDUrQouKIkePMIFHsL:307bc71b9ef91daaa1d77282446d1455864e5a3c56caa6743444f20d79467487"
        os.environ["ASTRA_DB_ID"] = "2f63f036-90c9-4ee7-b283-6df52c7231d5"
        os.environ["GROQ_API_KEY"] = "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" # Use empty key to test fallback routing
        
    def test_imports(self):
        """Verifies that all project submodules import successfully without raising syntax errors."""
        try:
            import app.config
            import app.database
            import app.ingest
            import app.agent_workflow
            import app.main
            imported = True
        except ImportError as e:
            imported = False
            print(f"Import Error: {e}")
        self.assertTrue(imported, "Failed to import all application modules cleanly.")

    def test_sqlite_db_init(self):
        """Verifies that the SQLite database is initialized with correct tables and schema."""
        from app.database import init_sqlite_db, TELEMETRY_DB_PATH
        
        # Initialise DB
        init_sqlite_db()
        self.assertTrue(TELEMETRY_DB_PATH.exists(), "SQLite database file was not created.")
        
        # Verify schema
        conn = sqlite3.connect(TELEMETRY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn("telemetry_logs", tables, "telemetry_logs table not found in SQLite DB.")
        self.assertIn("ingestion_history", tables, "ingestion_history table not found in SQLite DB.")
        
        conn.close()

    def test_routing_heuristics(self):
        """Verifies that the fallback routing heuristics correctly route keywords to their respective engines."""
        from app.agent_workflow import route_heuristic
        
        # Vector store questions
        self.assertEqual(route_heuristic("what is an agent?"), "vectorstore")
        self.assertEqual(route_heuristic("explain prompt engineering"), "vectorstore")
        self.assertEqual(route_heuristic("tell me about adversarial attacks on llms"), "vectorstore")
        
        # Wikipedia questions
        self.assertEqual(route_heuristic("who is Barack Obama?"), "wiki_search")
        self.assertEqual(route_heuristic("tell me about the history of Rome"), "wiki_search")

if __name__ == "__main__":
    unittest.main()
