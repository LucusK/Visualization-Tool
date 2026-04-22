import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "db" / "app.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows accessible as dicts
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't already exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT    NOT NULL,
                upload_time TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS passages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER NOT NULL REFERENCES documents(id),
                chunk_index INTEGER NOT NULL,
                chunk_text  TEXT    NOT NULL,
                emb_path    TEXT    NOT NULL
            );
        """)


# ── Documents ─────────────────────────────────────────────────────────────────

def insert_document(filename: str, upload_time: str) -> int:
    """Insert a document record and return its new id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, upload_time) VALUES (?, ?)",
            (filename, upload_time),
        )
        return cur.lastrowid


def get_document_name(doc_id: int) -> str:
    """Return the filename for a given document id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    return row["filename"] if row else "unknown"


# ── Passages ──────────────────────────────────────────────────────────────────

def insert_passage(doc_id: int, chunk_index: int, chunk_text: str, emb_path: str) -> int:
    """Insert a passage record and return its new id."""
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO passages (doc_id, chunk_index, chunk_text, emb_path)
               VALUES (?, ?, ?, ?)""",
            (doc_id, chunk_index, chunk_text, emb_path),
        )
        return cur.lastrowid


def get_all_passages() -> list[dict]:
    """Return all passages as a list of dicts with keys: id, doc_id, chunk_text, emb_path."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, doc_id, chunk_text, emb_path FROM passages ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_documents() -> list[dict]:
    """Return all documents as [{id, filename, upload_time}]."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, upload_time FROM documents ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_all() -> None:
    """Delete all documents and passages, reset autoincrement counters."""
    with _connect() as conn:
        conn.execute("DELETE FROM passages")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'passages')")
