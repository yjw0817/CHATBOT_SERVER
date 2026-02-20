"""MariaDB database setup and table creation."""
import os
import pymysql
from pymysql.cursors import DictCursor


def get_connection():
    """Get a MariaDB connection with DictCursor."""
    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "rag_db"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # apartments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS apartments (
            apt_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR(50) PRIMARY KEY,
            apt_id VARCHAR(50) NOT NULL,
            title VARCHAR(500),
            source_filename VARCHAR(500),
            source_type VARCHAR(20),
            content_hash VARCHAR(64),
            raw_text LONGTEXT,
            manualize_json LONGTEXT DEFAULT NULL,
            raw_text_hash VARCHAR(64) DEFAULT NULL,
            completed_phases VARCHAR(200) DEFAULT NULL,
            version INT DEFAULT 1,
            status VARCHAR(20) DEFAULT 'DRAFT',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id),
            INDEX idx_doc_apt (apt_id),
            INDEX idx_doc_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # doc_chunks (raw document split chunks for chunk-based manualize)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            chunk_id VARCHAR(50) PRIMARY KEY,
            doc_id VARCHAR(50) NOT NULL,
            chunk_index INT NOT NULL,
            start_anchor VARCHAR(500),
            end_anchor VARCHAR(500),
            char_start INT,
            char_end INT,
            split_basis VARCHAR(100),
            notes TEXT,
            raw_chunk LONGTEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            INDEX idx_dchunk_doc (doc_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # manual_sections
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS manual_sections (
            section_id VARCHAR(50) PRIMARY KEY,
            doc_id VARCHAR(50) NOT NULL,
            section_name VARCHAR(300),
            section_text LONGTEXT,
            ai_text LONGTEXT DEFAULT NULL,
            gate_status VARCHAR(20) DEFAULT NULL,
            gate_score SMALLINT DEFAULT NULL,
            gate_reasons_json TEXT DEFAULT NULL,
            prev_section_text LONGTEXT DEFAULT NULL,
            gate_stale TINYINT DEFAULT 0,
            source_chunk_id VARCHAR(50) DEFAULT NULL,
            evidence_json TEXT DEFAULT NULL,
            merge_status VARCHAR(20) DEFAULT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            INDEX idx_sec_doc (doc_id),
            INDEX idx_sec_gate (gate_status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # Migrate: add new columns if table already exists
    for table, col_def in [
        # manual_sections migrations
        ("manual_sections", ("gate_status", "VARCHAR(20) DEFAULT NULL")),
        ("manual_sections", ("gate_score", "SMALLINT DEFAULT NULL")),
        ("manual_sections", ("gate_reasons_json", "TEXT DEFAULT NULL")),
        ("manual_sections", ("prev_section_text", "LONGTEXT DEFAULT NULL")),
        ("manual_sections", ("gate_stale", "TINYINT DEFAULT 0")),
        ("manual_sections", ("updated_at", "DATETIME DEFAULT NULL")),
        ("manual_sections", ("source_chunk_id", "VARCHAR(50) DEFAULT NULL")),
        ("manual_sections", ("evidence_json", "TEXT DEFAULT NULL")),
        ("manual_sections", ("merge_status", "VARCHAR(20) DEFAULT NULL")),
        ("manual_sections", ("ai_text", "LONGTEXT DEFAULT NULL")),
        # documents migrations
        ("documents", ("manualize_json", "LONGTEXT DEFAULT NULL")),
        ("documents", ("raw_text_hash", "VARCHAR(64) DEFAULT NULL")),
        ("documents", ("completed_phases", "VARCHAR(200) DEFAULT NULL")),
    ]:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_def[0]} {col_def[1]}")
        except Exception:
            pass  # column already exists

    # manual_section_revisions (append-only history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS manual_section_revisions (
            revision_id VARCHAR(50) PRIMARY KEY,
            section_id VARCHAR(50) NOT NULL,
            doc_id VARCHAR(50) NOT NULL,
            version INT NOT NULL,
            section_text LONGTEXT,
            change_reason VARCHAR(30) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR(100) DEFAULT NULL,
            run_id VARCHAR(50) DEFAULT NULL,
            FOREIGN KEY (section_id) REFERENCES manual_sections(section_id),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            INDEX idx_rev_section (section_id),
            INDEX idx_rev_doc (doc_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # qa_issues
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_issues (
            issue_id VARCHAR(50) PRIMARY KEY,
            doc_id VARCHAR(50) NOT NULL,
            severity VARCHAR(10),
            issue_type VARCHAR(30),
            message TEXT,
            suggestion TEXT,
            status VARCHAR(20) DEFAULT 'OPEN',
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            INDEX idx_issue_doc (doc_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # chunks (for RAG)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR(50) PRIMARY KEY,
            doc_id VARCHAR(50) NOT NULL,
            section_name VARCHAR(300),
            chunk_index INT,
            chunk_text TEXT,
            embedding BLOB DEFAULT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            INDEX idx_chunk_doc (doc_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # conversations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id VARCHAR(50) PRIMARY KEY,
            apt_id VARCHAR(50),
            client_id VARCHAR(100),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id),
            INDEX idx_conv_apt (apt_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            msg_id VARCHAR(50) PRIMARY KEY,
            conversation_id VARCHAR(50) NOT NULL,
            role VARCHAR(20),
            text LONGTEXT,
            meta_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            INDEX idx_msg_conv (conversation_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # improve_suggestions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS improve_suggestions (
            sug_id VARCHAR(50) PRIMARY KEY,
            apt_id VARCHAR(50),
            title VARCHAR(500),
            reason TEXT,
            proposed_patch TEXT,
            target_doc_id VARCHAR(50),
            target_section_name VARCHAR(300),
            status VARCHAR(20) DEFAULT 'PENDING',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # api_specs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_specs (
            spec_id VARCHAR(50) PRIMARY KEY,
            doc_id VARCHAR(50) NOT NULL,
            intent VARCHAR(300),
            endpoint VARCHAR(300),
            method VARCHAR(10),
            req_fields_json TEXT,
            resp_fields_json TEXT,
            auth VARCHAR(200),
            errors_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
            INDEX idx_spec_doc (doc_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # branch_class_cache (Step 4-1)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS branch_class_cache (
            apt_id VARCHAR(50) NOT NULL,
            branch_id VARCHAR(50) NOT NULL,
            class_id VARCHAR(50) NOT NULL,
            name VARCHAR(200),
            start VARCHAR(50),
            end VARCHAR(50),
            capacity INT,
            reserved INT,
            asof VARCHAR(50),
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (apt_id, branch_id, class_id),
            FOREIGN KEY (apt_id) REFERENCES apartments(apt_id),
            INDEX idx_bcc_branch (branch_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at MariaDB {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
