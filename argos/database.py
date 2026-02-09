"""SQLite database setup and table creation."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "argos.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # apartments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS apartments (
            apt_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            apt_id TEXT NOT NULL,
            title TEXT,
            source_filename TEXT,
            source_type TEXT,
            content_hash TEXT,
            raw_text TEXT,
            version INTEGER DEFAULT 1,
            status TEXT DEFAULT 'DRAFT',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id)
        )
    """)
    
    # manual_sections
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS manual_sections (
            section_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            section_name TEXT,
            section_text TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)
    
    # qa_issues
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_issues (
            issue_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            severity TEXT,
            issue_type TEXT,
            message TEXT,
            suggestion TEXT,
            status TEXT DEFAULT 'OPEN',
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)
    
    # chunks (for RAG)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            section_name TEXT,
            chunk_index INTEGER,
            chunk_text TEXT,
            embedding TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)
    
    # conversations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            apt_id TEXT,
            client_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id)
        )
    """)
    
    # messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            msg_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT,
            text TEXT,
            meta_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    """)
    
    # improve_suggestions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS improve_suggestions (
            sug_id TEXT PRIMARY KEY,
            apt_id TEXT,
            title TEXT,
            reason TEXT,
            proposed_patch TEXT,
            target_doc_id TEXT,
            target_section_name TEXT,
            status TEXT DEFAULT 'PENDING',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id)
        )
    """)
    
    # api_specs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_specs (
            spec_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            intent TEXT,
            endpoint TEXT,
            method TEXT,
            req_fields_json TEXT,
            resp_fields_json TEXT,
            auth TEXT,
            errors_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)
    
    # branch_class_cache (Step 4-1)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS branch_class_cache (
            apt_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            class_id TEXT NOT NULL,
            name TEXT,
            start TEXT,
            end TEXT,
            capacity INTEGER,
            reserved INTEGER,
            asof TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (apt_id, branch_id, class_id),
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")
