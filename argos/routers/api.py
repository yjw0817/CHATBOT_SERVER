"""API routes for apartments, documents, and file upload."""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from database import get_connection

router = APIRouter(prefix="/api", tags=["api"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Import unified LLM client
from llm_client import call_llm, is_llm_available, get_llm_info, LLM_MODEL

@router.get("/debug/llm")
def debug_llm():
    """Debug endpoint to check LLM config."""
    import os
    return {
        "is_llm_available": is_llm_available(),
        "get_llm_info": get_llm_info(),
        "env_LLM_ENABLED": os.getenv("LLM_ENABLED"),
        "env_LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
        "env_LLM_API_KEY_set": bool(os.getenv("LLM_API_KEY"))
    }


class ApartmentCreate(BaseModel):
    apt_id: str
    name: str


class ApartmentResponse(BaseModel):
    apt_id: str
    name: str
    created_at: str


# ============ APARTMENTS ============

@router.post("/apartments")
def create_apartment(apt_id: str = Form(...), name: str = Form(...)):
    """Create a new apartment (accepts form data)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO apartments (apt_id, name) VALUES (?, ?)",
            (apt_id, name)
        )
        conn.commit()
        return {"success": True, "apt_id": apt_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@router.get("/apartments")
def list_apartments():
    """List all apartments."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT apt_id, name, created_at FROM apartments ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ============ UPLOAD ============

@router.post("/upload")
async def upload_document(
    apt_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a document file."""
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()
    
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    source_type = ext if ext in ("docx", "pdf", "txt", "md") else "txt"
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT doc_id, version FROM documents WHERE apt_id = ? AND content_hash = ? AND status != 'ARCHIVED' ORDER BY version DESC LIMIT 1",
        (apt_id, content_hash)
    )
    existing = cursor.fetchone()
    
    if existing:
        new_version = existing["version"] + 1
        cursor.execute(
            "UPDATE documents SET status = 'ARCHIVED', updated_at = ? WHERE apt_id = ? AND content_hash = ? AND status != 'ARCHIVED'",
            (datetime.now().isoformat(), apt_id, content_hash)
        )
    else:
        cursor.execute("SELECT MAX(version) as max_ver FROM documents WHERE apt_id = ?", (apt_id,))
        row = cursor.fetchone()
        new_version = (row["max_ver"] or 0) + 1
    
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    file_path = UPLOAD_DIR / f"{doc_id}.{source_type}"
    with open(file_path, "wb") as f:
        f.write(content)
    
    raw_text = f"[Placeholder: extract text first]"
    
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO documents (doc_id, apt_id, title, source_filename, source_type, content_hash, raw_text, version, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT', ?, ?)
    """, (doc_id, apt_id, filename, filename, source_type, content_hash, raw_text, new_version, now, now))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "doc_id": doc_id, "version": new_version, "content_hash": content_hash, "status": "DRAFT"}


@router.get("/docs")
def list_documents(apt_id: Optional[str] = None):
    """List documents, optionally filtered by apartment."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if apt_id:
        cursor.execute(
            "SELECT doc_id, apt_id, title, source_filename, source_type, version, status, created_at FROM documents WHERE apt_id = ? ORDER BY created_at DESC",
            (apt_id,)
        )
    else:
        cursor.execute(
            "SELECT doc_id, apt_id, title, source_filename, source_type, version, status, created_at FROM documents ORDER BY created_at DESC"
        )
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ============ STEP 2: EXTRACT TEXT ============

@router.post("/doc/{doc_id}/extract-text")
def extract_text(doc_id: str):
    """Extract text from uploaded document (DOCX supported)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT source_filename, source_type FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    source_type = doc["source_type"]
    file_path = UPLOAD_DIR / f"{doc_id}.{source_type}"
    
    if not file_path.exists():
        conn.close()
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    raw_text = ""
    
    if source_type == "docx":
        try:
            from docx import Document
            docx_doc = Document(str(file_path))
            paragraphs = [p.text for p in docx_doc.paragraphs if p.text.strip()]
            raw_text = "\n".join(paragraphs)
        except Exception as e:
            conn.close()
            raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {str(e)}")
    elif source_type in ("txt", "md"):
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raw_text = f"[PDF extraction not implemented - file: {doc['source_filename']}]"
    
    cursor.execute("UPDATE documents SET raw_text = ?, updated_at = ? WHERE doc_id = ?",
                   (raw_text, datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "doc_id": doc_id,
        "chars": len(raw_text),
        "preview": raw_text[:300] if raw_text else ""
    }


# ============ STEP 2: MANUALIZE ============

MANUALIZE_PROMPT = """당신은 업무 문서를 구조화된 매뉴얼로 변환하는 전문가입니다.

아래 원본 텍스트를 분석하여:
1) 문서 내용에 맞는 핵심 섹션들을 자동으로 도출하고 내용을 bullet point로 정리하세요.
2) 문서 내에서 모호하거나, 누락되었거나, 추가 확인이 필요한 사항들을 찾아 'todo_questions'로 리스트업 하세요.

반환 형식: JSON 객체
{{
  "sections": {{
    "섹션명1": "- 내용...",
    "섹션명2": "- 내용..."
  }},
  "todo_questions": ["질문1", "질문2", ...]
}}

JSON만 반환하세요. 다른 텍스트 없이."""


@router.post("/doc/{doc_id}/manualize")
def manualize(doc_id: str, force: bool = False):
    """Convert raw text to derived manual sections and generate AI todo questions."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = doc["raw_text"]
    if not raw_text or raw_text.startswith("[Placeholder"):
        conn.close()
        raise HTTPException(status_code=400, detail="Extract text first")

    # Check if manual sections already exist (return cached if not forced)
    if not force:
        cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = ?", (doc_id,))
        existing = cursor.fetchall()
        if existing:
            # We also need todo_questions. For now, let's assume if sections exist, we might have some stored or just return sections.
            conn.close()
            return {
                "success": True,
                "doc_id": doc_id,
                "sections": [row["section_name"] for row in existing],
                "section_details": {row["section_name"]: row["section_text"] for row in existing},
                "llm_used": False,
                "cached": True,
                "todo_questions": [] # In cache mode, we skip new questions unless forced
            }

    # Use LLM or fallback
    result_data = {}
    llm_error_msg = None

    llm_available = is_llm_available()

    if llm_available:
        try:
            content = call_llm(MANUALIZE_PROMPT.format(raw_text=raw_text[:8000]), temperature=0.3)
            if content:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result_data = json.loads(json_match.group())
            else:
                llm_error_msg = "LLM 응답이 비어있습니다."
        except Exception as e:
            llm_error_msg = f"LLM 호출 실패: {str(e)}"

    sections = result_data.get("sections", {})
    todo_questions = result_data.get("todo_questions", [])

    if not sections:
        # Fallback: extract headings
        lines = raw_text.split("\n")
        current_section = "일반"
        section_lines = {}
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                current_section = stripped.lstrip("#").strip()[:30]
                if current_section not in section_lines: section_lines[current_section] = []
            elif stripped:
                if current_section not in section_lines: section_lines[current_section] = []
                section_lines[current_section].append(f"- {stripped}")
        for key, val_lines in section_lines.items():
            sections[key] = "\n".join(val_lines[:20])
        if not sections:
            sections["전체 내용"] = raw_text[:2000]
    
    # Save sections
    cursor.execute("DELETE FROM manual_sections WHERE doc_id = ?", (doc_id,))
    for section_name, section_text in sections.items():
        section_id = f"sec_{uuid.uuid4().hex[:8]}"
        cursor.execute(
            "INSERT INTO manual_sections (section_id, doc_id, section_name, section_text) VALUES (?, ?, ?, ?)",
            (section_id, doc_id, section_name, section_text if section_text else "정보 없음")
        )
    
    cursor.execute("UPDATE documents SET updated_at = ? WHERE doc_id = ?", (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "doc_id": doc_id,
        "sections": list(sections.keys()),
        "section_details": sections,
        "todo_questions": todo_questions,
        "llm_used": bool(sections and llm_available and not llm_error_msg),
        "llm_error": llm_error_msg
    }


# ============ STEP 2: QUALITY GATE ============

QUALITY_GATE_PROMPT = """당신은 아파트 운영 매뉴얼의 품질을 검증하는 전문가입니다.

아래 매뉴얼 섹션들을 검토하고 이슈를 JSON 배열로 반환하세요.

이슈 타입:
- MISSING (RED): 환불/예약/운영시간/권한 중 핵심 정보가 완전히 없음
- AMBIGUOUS (YELLOW): "상황에 따라", "가능하면", "적당히", "협의 후" 등 모호한 표현
- CONFLICT (RED): 같은 주제에서 상반된 규칙 발견
- PII_RISK (RED): 주민번호/전화번호 등 개인정보 패턴
- API_NEEDED (YELLOW): "예약 생성", "문자 발송", "강좌 추가" 등 시스템 연동 필요

각 이슈 형식:
{{"severity": "RED|YELLOW", "issue_type": "타입", "message": "설명", "suggestion": "해결방안"}}

매뉴얼 내용:
{sections_text}

JSON 배열만 반환하세요."""


@router.post("/doc/{doc_id}/quality-gate")
def quality_gate(doc_id: str):
    """Run quality checks on manual sections."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = ?", (doc_id,))
    sections = cursor.fetchall()
    if not sections:
        conn.close()
        raise HTTPException(status_code=400, detail="Manualize first")
    
    sections_text = "\n\n".join([f"[{s['section_name']}]\n{s['section_text']}" for s in sections])
    
    issues = []
    
    # Rule-based checks first
    # PII check
    pii_patterns = [
        (r'\d{6}-\d{7}', '주민번호'),
        (r'010-?\d{4}-?\d{4}', '전화번호'),
    ]
    for pattern, pii_type in pii_patterns:
        if re.search(pattern, sections_text):
            issues.append({
                "severity": "RED",
                "issue_type": "PII_RISK",
                "message": f"{pii_type} 패턴이 발견되었습니다",
                "suggestion": "개인정보를 삭제하거나 마스킹하세요"
            })
    
    # Ambiguous phrases check
    ambiguous_phrases = ["상황에 따라", "가능하면", "적당히", "협의 후", "경우에 따라", "필요시"]
    for phrase in ambiguous_phrases:
        if phrase in sections_text:
            issues.append({
                "severity": "YELLOW",
                "issue_type": "AMBIGUOUS",
                "message": f"모호한 표현 발견: '{phrase}'",
                "suggestion": "구체적인 기준이나 조건으로 명시하세요"
            })
    
    # API needed check
    api_phrases = ["예약 생성", "예약 취소", "문자 발송", "SMS", "강좌 추가", "강좌 삭제", "회원 등록"]
    for phrase in api_phrases:
        if phrase in sections_text:
            issues.append({
                "severity": "YELLOW",
                "issue_type": "API_NEEDED",
                "message": f"시스템 연동 필요: '{phrase}'",
                "suggestion": "해당 기능의 API 스펙을 정의하세요"
            })
    
    # Missing check for critical sections
    for s in sections:
        if s["section_text"] == "정보 없음" and s["section_name"] in ["환불/위약/정산", "예약/취소/변경", "운영시간/휴무"]:
            issues.append({
                "severity": "RED",
                "issue_type": "MISSING",
                "message": f"필수 섹션 '{s['section_name']}'의 내용이 없습니다",
                "suggestion": "해당 규정을 추가하세요"
            })
    
    # LLM-based additional checks if available
    llm_error_msg = None
    if is_llm_available() and len(issues) < 5:
        try:
            content = call_llm(QUALITY_GATE_PROMPT.format(sections_text=sections_text[:6000]), temperature=0.3)
            if content:
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    llm_issues = json.loads(json_match.group())
                    issues.extend(llm_issues[:5])  # Limit LLM issues
            else:
                llm_error_msg = "LLM 응답이 비어있습니다. (API 할당량 초과 가능성)"
        except Exception as e:
            llm_error_msg = f"LLM 호출 실패: {str(e)}"
            print(f"[QUALITY_GATE] LLM error: {e}")
    
    # Clear old issues and save new
    cursor.execute("DELETE FROM qa_issues WHERE doc_id = ?", (doc_id,))
    for issue in issues:
        issue_id = f"issue_{uuid.uuid4().hex[:8]}"
        cursor.execute(
            "INSERT INTO qa_issues (issue_id, doc_id, severity, issue_type, message, suggestion, status) VALUES (?, ?, ?, ?, ?, ?, 'OPEN')",
            (issue_id, doc_id, issue.get("severity", "YELLOW"), issue.get("issue_type", "OTHER"), 
             issue.get("message", ""), issue.get("suggestion", ""))
        )
    
    conn.commit()
    conn.close()
    
    red_count = len([i for i in issues if i.get("severity") == "RED"])
    yellow_count = len([i for i in issues if i.get("severity") == "YELLOW"])
    
    return {
        "success": True, 
        "doc_id": doc_id, 
        "red_count": red_count, 
        "yellow_count": yellow_count, 
        "issues": issues,
        "llm_error": llm_error_msg,
        "api_specs": extract_api_spec(doc_id).get("specs", [])
    }


# ============ STEP 2: UPDATE SECTIONS ============

class SectionsUpdate(BaseModel):
    sections: dict # {section_name: text}

@router.put("/doc/{doc_id}/sections")
def update_sections(doc_id: str, req: SectionsUpdate):
    """Update manual sections manually."""
    conn = get_connection()
    cursor = conn.cursor()

    for name, text in req.sections.items():
        cursor.execute(
            "UPDATE manual_sections SET section_text = ? WHERE doc_id = ? AND section_name = ?",
            (text, doc_id, name)
        )
    
    cursor.execute("UPDATE documents SET updated_at = ? WHERE doc_id = ?", (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()
    
    return {"success": True, "doc_id": doc_id}


# ============ STEP 2: AI HELPER (Refine/Fill/Recommend) ============

class RefineRequest(BaseModel):
    text: str
    task: str  # "refine", "fill", "recommend"
    context: Optional[str] = None # Section name or issue message

REFINE_PROMPT = """당신은 매뉴얼 전문 에디터입니다. 다음 요청을 수행하세요.

[요청 타입]: {task}
[대상 텍스트]: {text}
[추가 컨텍스트]: {context}

[작업 지침]
- "refine": 선택한 문장을 더 명확하고 전문적인 어조로 다듬어주세요.
- "fill": 부족한 정보나 질문에 대해 보강할 수 있는 적절한 문장을 제안하세요. (모르면 '확인 필요' 명시)
- "recommend": 해당 섹션에 어울리는 표준적인 매뉴얼 문구나 템플릿을 추천하세요.

반환 형식: 제안하는 텍스트만 출력하세요. 다른 설명은 생략하세요."""

@router.post("/doc/{doc_id}/refine-text")
def refine_text(doc_id: str, req: RefineRequest):
    """AI helper for specific editing tasks."""
    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM 사용 불가")
    
    try:
        suggestion = call_llm(
            REFINE_PROMPT.format(task=req.task, text=req.text, context=req.context or ""),
            temperature=0.3
        )
        return {"success": True, "suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ STEP 2: APPROVE ============

@router.post("/doc/{doc_id}/approve")
def approve(doc_id: str):
    """Approve document if no RED issues open."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    cursor.execute("SELECT COUNT(*) as cnt FROM qa_issues WHERE doc_id = ? AND severity = 'RED' AND status = 'OPEN'", (doc_id,))
    red_count = cursor.fetchone()["cnt"]
    
    if red_count > 0:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Cannot approve: {red_count} RED issues open")
    
    cursor.execute("UPDATE documents SET status = 'APPROVED', updated_at = ? WHERE doc_id = ?",
                   (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()
    
    # Auto-reindex after approval
    reindex_result = reindex(doc_id)
    
    return {"success": True, "doc_id": doc_id, "status": "APPROVED", "reindex": reindex_result}


# ============ STEP 2: REINDEX ============

@router.post("/doc/{doc_id}/reindex")
def reindex(doc_id: str):
    """Chunk manual sections for RAG retrieval."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = ?", (doc_id,))
    sections = cursor.fetchall()
    if not sections:
        conn.close()
        raise HTTPException(status_code=400, detail="No sections to index")
    
    # Delete existing chunks
    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    
    chunk_size = 500
    overlap = 100
    chunk_count = 0
    
    for section in sections:
        text = section["section_text"]
        section_name = section["section_name"]
        
        if not text or text == "정보 없음":
            continue
        
        # Simple chunking with overlap
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
                cursor.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, section_name, chunk_index, chunk_text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, section_name, chunk_index, chunk_text, datetime.now().isoformat())
                )
                chunk_count += 1
                chunk_index += 1
            
            start = end - overlap
            if start >= len(text):
                break
    
    conn.commit()
    conn.close()
    
    return {"success": True, "doc_id": doc_id, "chunk_count": chunk_count}


# ============ STEP 2: GET SECTIONS ============

@router.get("/doc/{doc_id}/sections")
def get_sections(doc_id: str):
    """Get manual sections for a document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = ?", (doc_id,))
    sections = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sections


@router.get("/doc/{doc_id}/issues")
def get_issues(doc_id: str):
    """Get QA issues for a document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT issue_id, severity, issue_type, message, suggestion, status FROM qa_issues WHERE doc_id = ?", (doc_id,))
    issues = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return issues


# ============ STEP 3: CHAT WITH RAG ============

class ChatRequest(BaseModel):
    apt_id: str
    client_id: str = "default"
    conversation_id: Optional[str] = None
    message: str


CHAT_PROMPT = """당신은 아파트 커뮤니티 규정에 대해 답변하는 AI 어시스턴트입니다.

아래 문서 내용을 바탕으로 사용자 질문에 답변하세요.
문서에 없는 내용은 답변하지 마세요. 반드시 아래 JSON 형식으로만 응답하세요.

문서 내용:
{context}

사용자 질문: {question}

JSON 형식:
{{
  "reply_text": "답변 내용 (간결하게)",
  "citations": [
    {{"doc_id": "문서ID", "doc_title": "문서명", "section_name": "섹션명", "snippet": "인용 부분 120자 이내"}}
  ],
  "confidence": "HIGH|MED|LOW",
  "next_question": null 또는 "추가 질문(근거 부족시)",
  "actions": []
}}

규칙:
- citations이 없으면 confidence는 LOW로 설정
- LOW일 경우 next_question에 확인할 질문 1개 포함
- 예약생성/문자발송 등 실행 액션은 금지 (actions는 항상 [])
- 근거 없이 추측하지 마세요"""


def keyword_search(query: str, chunks: list, top_k: int = 5) -> list:
    """Simple keyword-based search scoring."""
    keywords = set(query.lower().replace("?", "").replace(".", "").split())
    scored = []
    for chunk in chunks:
        text = chunk["chunk_text"].lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


@router.post("/chat")
def chat(req: ChatRequest):
    """Chat with RAG retrieval and citations."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get or create conversation
    if req.conversation_id:
        conv_id = req.conversation_id
        cursor.execute("UPDATE conversations SET last_at = ? WHERE conversation_id = ?",
                       (datetime.now().isoformat(), conv_id))
    else:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        cursor.execute(
            "INSERT INTO conversations (conversation_id, apt_id, client_id, created_at, last_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, req.apt_id, req.client_id, datetime.now().isoformat(), datetime.now().isoformat())
        )
    
    # Save user message
    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    cursor.execute(
        "INSERT INTO messages (msg_id, conversation_id, role, text, created_at) VALUES (?, ?, 'user', ?, ?)",
        (user_msg_id, conv_id, req.message, datetime.now().isoformat())
    )
    
    # Retrieve chunks from APPROVED documents for this apt_id
    cursor.execute("""
        SELECT c.chunk_id, c.doc_id, c.section_name, c.chunk_text, d.title as doc_title
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE d.apt_id = ? AND d.status = 'APPROVED'
    """, (req.apt_id,))
    all_chunks = [dict(row) for row in cursor.fetchall()]
    
    # Keyword search
    top_chunks = keyword_search(req.message, all_chunks, top_k=5)
    
    # Default response
    response = {
        "conversation_id": conv_id,
        "reply_text": "",
        "citations": [],
        "confidence": "LOW",
        "next_question": None,
        "actions": []
    }
    
    if not top_chunks:
        response["reply_text"] = "죄송합니다. 관련 정보를 찾을 수 없습니다."
        response["next_question"] = "어떤 내용에 대해 더 알고 싶으신가요?"
    else:
        # Build context
        context_parts = []
        for chunk in top_chunks:
            context_parts.append(f"[문서: {chunk['doc_title']} / 섹션: {chunk['section_name']}]\n{chunk['chunk_text']}")
        context = "\n\n".join(context_parts)
        
        if is_llm_available():
            try:
                content = call_llm(CHAT_PROMPT.format(context=context, question=req.message), temperature=0.3)
                if content:
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        response["reply_text"] = parsed.get("reply_text", "")
                        response["citations"] = parsed.get("citations", [])
                        response["confidence"] = parsed.get("confidence", "MED")
                        response["next_question"] = parsed.get("next_question")
            except Exception as e:
                response["reply_text"] = f"LLM 오류: {str(e)[:50]}"
        
        if not response["reply_text"]:
            # Mock response without LLM
            response["reply_text"] = f"문서에서 {len(top_chunks)}개의 관련 정보를 찾았습니다."
            response["citations"] = [
                {
                    "doc_id": c["doc_id"],
                    "doc_title": c["doc_title"],
                    "section_name": c["section_name"],
                    "snippet": c["chunk_text"][:150] + "..." if len(c["chunk_text"]) > 150 else c["chunk_text"]
                } for c in top_chunks[:2]
            ]
            response["confidence"] = "MED" if top_chunks else "LOW"
    
    # Save assistant message
    asst_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    meta_json = json.dumps({
        "citations": response["citations"],
        "confidence": response["confidence"],
        "retrieval_count": len(top_chunks)
    }, ensure_ascii=False)
    cursor.execute(
        "INSERT INTO messages (msg_id, conversation_id, role, text, meta_json, created_at) VALUES (?, ?, 'assistant', ?, ?, ?)",
        (asst_msg_id, conv_id, response["reply_text"], meta_json, datetime.now().isoformat())
    )
    
    conn.commit()
    conn.close()
    
    return response


# ============ STEP 3: IMPROVEMENTS GENERATOR ============

@router.post("/improvements/generate")
def generate_improvements(apt_id: str):
    """Generate improvement suggestions from chat logs."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get recent assistant messages with LOW confidence or empty citations
    cursor.execute("""
        SELECT m.msg_id, m.text, m.meta_json, m.created_at, c.apt_id
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.apt_id = ? AND m.role = 'assistant'
        ORDER BY m.created_at DESC
        LIMIT 50
    """, (apt_id,))
    messages = cursor.fetchall()
    
    suggestions = []
    seen_topics = set()
    
    for msg in messages:
        meta = json.loads(msg["meta_json"] or "{}")
        confidence = meta.get("confidence", "HIGH")
        citations = meta.get("citations", [])
        
        # Check for LOW confidence or empty citations
        if confidence == "LOW" or not citations:
            # Extract topic from message text
            text = msg["text"][:100]
            topic_key = text[:30]
            
            if topic_key not in seen_topics and len(suggestions) < 5:
                seen_topics.add(topic_key)
                
                # Determine target section
                target_section = "예외/문의/권한"
                if "환불" in text or "정산" in text:
                    target_section = "환불/위약/정산"
                elif "예약" in text or "취소" in text:
                    target_section = "예약/취소/변경"
                elif "운영" in text or "시간" in text:
                    target_section = "운영시간/휴무"
                
                suggestions.append({
                    "title": f"정보 보완 필요: {text[:30]}...",
                    "reason": f"confidence={confidence}, citations={len(citations)}개",
                    "target_section_name": target_section,
                    "proposed_patch": ""
                })
    
    # Get the latest APPROVED doc for this apt
    cursor.execute(
        "SELECT doc_id FROM documents WHERE apt_id = ? AND status = 'APPROVED' ORDER BY version DESC LIMIT 1",
        (apt_id,)
    )
    doc_row = cursor.fetchone()
    target_doc_id = doc_row["doc_id"] if doc_row else None
    
    # Save suggestions
    for sug in suggestions:
        sug_id = f"sug_{uuid.uuid4().hex[:8]}"
        cursor.execute("""
            INSERT INTO improve_suggestions (sug_id, apt_id, title, reason, proposed_patch, target_doc_id, target_section_name, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
        """, (sug_id, apt_id, sug["title"], sug["reason"], sug["proposed_patch"], target_doc_id, sug["target_section_name"],
              datetime.now().isoformat(), datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "count": len(suggestions), "suggestions": suggestions}


# ============ STEP 3: ONE-CLICK PATCH APPLY ============

PATCH_PROMPT = """문서 섹션에 추가할 FAQ 항목을 생성하세요.

제안 제목: {title}
제안 이유: {reason}
대상 섹션: {section_name}

현재 섹션 내용:
{section_text}

규칙:
- 없는 규정을 만들지 마세요
- 확실하지 않으면 "확인 필요" 형태로 작성
- 간결한 Q&A 형식으로 작성

출력 형식 (추가할 텍스트만):
---FAQ---
Q: 질문
A: 답변
---"""


@router.post("/improvements/{sug_id}/apply")
def apply_improvement(sug_id: str):
    """Apply improvement suggestion with one-click patch."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get suggestion
    cursor.execute("SELECT * FROM improve_suggestions WHERE sug_id = ?", (sug_id,))
    sug = cursor.fetchone()
    if not sug:
        conn.close()
        raise HTTPException(status_code=404, detail="Suggestion not found")
    
    if sug["status"] == "APPLIED":
        conn.close()
        raise HTTPException(status_code=400, detail="Already applied")
    
    target_doc_id = sug["target_doc_id"]
    target_section = sug["target_section_name"]
    
    # Get current section text
    cursor.execute(
        "SELECT section_id, section_text FROM manual_sections WHERE doc_id = ? AND section_name = ?",
        (target_doc_id, target_section)
    )
    section_row = cursor.fetchone()
    
    if not section_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Target section not found")
    
    current_text = section_row["section_text"]
    section_id = section_row["section_id"]
    
    # Generate patch
    patch_text = ""
    if is_llm_available():
        try:
            prompt = PATCH_PROMPT.format(
                title=sug["title"],
                reason=sug["reason"],
                section_name=target_section,
                section_text=current_text[:2000]
            )
            patch_text = call_llm(prompt, temperature=0.3)
        except Exception as e:
            print(f"[PATCH] Error: {e}")
            patch_text = f"\n\n---FAQ---\nQ: {sug['title']}\nA: 확인 필요 - 관리자에게 문의하세요."
    
    if not patch_text:
        patch_text = f"\n\n---FAQ---\nQ: {sug['title']}\nA: 확인 필요 - 관리자에게 문의하세요."
    
    # Append patch to section
    new_text = current_text + "\n" + patch_text
    cursor.execute(
        "UPDATE manual_sections SET section_text = ? WHERE section_id = ?",
        (new_text, section_id)
    )
    
    # Update suggestion status
    cursor.execute(
        "UPDATE improve_suggestions SET status = 'APPLIED', updated_at = ? WHERE sug_id = ?",
        (datetime.now().isoformat(), sug_id)
    )
    
    conn.commit()
    conn.close()
    
    # Reindex the document
    reindex_result = reindex(target_doc_id)
    
    return {"success": True, "sug_id": sug_id, "status": "APPLIED", "reindexed": reindex_result}


# ============ STEP 3: API SPEC EXTRACTOR ============

API_SPEC_PROMPT = """아래 매뉴얼 섹션에서 시스템 API가 필요한 의도(intent)를 추출하세요.

매뉴얼 내용:
{sections_text}

품질 이슈 (API_NEEDED):
{api_issues}

각 intent에 대해 API 스펙을 JSON 배열로 반환하세요:
[
  {{
    "intent_name": "예약 생성",
    "endpoint": "/api/booking/create",
    "method": "POST",
    "request_fields": ["member_id", "class_id", "date"],
    "response_fields": ["booking_id", "status"],
    "auth": "입주민|관리자|시스템",
    "notes": "비고"
  }}
]

JSON 배열만 반환하세요."""


@router.post("/doc/{doc_id}/extract-api-spec")
def extract_api_spec(doc_id: str):
    """Extract API specifications from document."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get sections
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = ?", (doc_id,))
    sections = cursor.fetchall()
    
    # Get API_NEEDED issues
    cursor.execute("SELECT message FROM qa_issues WHERE doc_id = ? AND issue_type = 'API_NEEDED'", (doc_id,))
    api_issues = [row["message"] for row in cursor.fetchall()]
    
    sections_text = "\n\n".join([f"[{s['section_name']}]\n{s['section_text']}" for s in sections])
    
    specs = []
    
    if is_llm_available():
        try:
            prompt = API_SPEC_PROMPT.format(
                sections_text=sections_text[:4000],
                api_issues=", ".join(api_issues)
            )
            content = call_llm(prompt, temperature=0.3)
            if content:
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    specs = json.loads(json_match.group())
        except Exception as e:
            print(f"[API_SPEC] Error: {e}")
    
    # Fallback: generate from API_NEEDED issues
    if not specs:
        for issue in api_issues:
            intent = issue.replace("시스템 연동 필요: ", "").replace("'", "")
            specs.append({
                "intent_name": intent,
                "endpoint": f"/api/{intent.replace(' ', '-').lower()}",
                "method": "POST",
                "request_fields": ["member_id"],
                "response_fields": ["status", "message"],
                "auth": "관리자",
                "notes": "자동 추출됨 - 검토 필요"
            })
    
    # Clear and save specs
    cursor.execute("DELETE FROM api_specs WHERE doc_id = ?", (doc_id,))
    for spec in specs:
        spec_id = f"spec_{uuid.uuid4().hex[:8]}"
        cursor.execute("""
            INSERT INTO api_specs (spec_id, doc_id, intent, endpoint, method, req_fields_json, resp_fields_json, auth, errors_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (spec_id, doc_id, spec.get("intent_name", ""), spec.get("endpoint", ""), spec.get("method", "POST"),
              json.dumps(spec.get("request_fields", []), ensure_ascii=False),
              json.dumps(spec.get("response_fields", []), ensure_ascii=False),
              spec.get("auth", ""), "[]", datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "doc_id": doc_id, "spec_count": len(specs), "specs": specs}


@router.get("/doc/{doc_id}/api-spec/export")
def export_api_spec(doc_id: str, format: str = "json"):
    """Export API specs as JSON or YAML."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT intent, endpoint, method, req_fields_json, resp_fields_json, auth, errors_json
        FROM api_specs WHERE doc_id = ?
    """, (doc_id,))
    rows = cursor.fetchall()
    conn.close()
    
    specs = []
    for row in rows:
        specs.append({
            "intent": row["intent"],
            "endpoint": row["endpoint"],
            "method": row["method"],
            "request_fields": json.loads(row["req_fields_json"] or "[]"),
            "response_fields": json.loads(row["resp_fields_json"] or "[]"),
            "auth": row["auth"],
            "errors": json.loads(row["errors_json"] or "[]")
        })
    
    if format == "yaml":
        # Simple YAML conversion
        yaml_lines = ["api_specs:"]
        for spec in specs:
            yaml_lines.append(f"  - intent: {spec['intent']}")
            yaml_lines.append(f"    endpoint: {spec['endpoint']}")
            yaml_lines.append(f"    method: {spec['method']}")
            yaml_lines.append(f"    auth: {spec['auth']}")
            yaml_lines.append(f"    request_fields: {spec['request_fields']}")
            yaml_lines.append(f"    response_fields: {spec['response_fields']}")
        return {"format": "yaml", "content": "\n".join(yaml_lines)}
    
    return {"format": "json", "specs": specs}


# ============ STEP 3: GET SUGGESTIONS ============

@router.get("/improvements")
def list_improvements(apt_id: Optional[str] = None):
    """List improvement suggestions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if apt_id:
        cursor.execute("""
            SELECT s.*, a.name as apt_name
            FROM improve_suggestions s
            LEFT JOIN apartments a ON s.apt_id = a.apt_id
            WHERE s.apt_id = ?
            ORDER BY s.created_at DESC
        """, (apt_id,))
    else:
        cursor.execute("""
            SELECT s.*, a.name as apt_name
            FROM improve_suggestions s
            LEFT JOIN apartments a ON s.apt_id = a.apt_id
            ORDER BY s.created_at DESC
        """)
    
    suggestions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return suggestions


@router.get("/conversations")
def list_conversations(apt_id: Optional[str] = None):
    """List conversations."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if apt_id:
        cursor.execute("SELECT * FROM conversations WHERE apt_id = ? ORDER BY last_at DESC", (apt_id,))
    else:
        cursor.execute("SELECT * FROM conversations ORDER BY last_at DESC")
    
    convs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return convs


@router.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str):
    """Get messages for a conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at", (conversation_id,))
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages


# ============ STEP 4-1: BRANCH CLASS SYNC ============

@router.post("/branch/{branch_id}/classes/sync")
def sync_branch_classes(
    branch_id: str,
    apt_id: str = Form(...),
    classes_json: str = Form(...),
    asof: Optional[str] = Form(None)
):
    """Sync branch classes from form-urlencoded data.
    
    Accepts application/x-www-form-urlencoded with:
    - apt_id: apartment ID (required)
    - classes_json: JSON string array of class objects (required)
    - asof: timestamp string (optional)
    """
    # Parse classes_json
    try:
        classes = json.loads(classes_json)
        if not isinstance(classes, list):
            raise ValueError("classes_json must be a JSON array")
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": str(e),
            "preview": classes_json[:100] if classes_json else "",
            "hint": "classes_json must be valid JSON array"
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "preview": classes_json[:100] if classes_json else "",
            "hint": "classes_json must be valid JSON array"
        }
    
    conn = get_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    asof_value = asof or now
    upserted = 0
    
    for cls in classes:
        class_id = cls.get("class_id") or cls.get("id") or f"cls_{uuid.uuid4().hex[:8]}"
        name = cls.get("name", "")
        start = cls.get("start", "")
        end = cls.get("end", "")
        capacity = cls.get("capacity", 0)
        reserved = cls.get("reserved", 0)
        
        # Upsert using INSERT OR REPLACE
        cursor.execute("""
            INSERT OR REPLACE INTO branch_class_cache 
            (apt_id, branch_id, class_id, name, start, end, capacity, reserved, asof, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (apt_id, branch_id, class_id, name, start, end, capacity, reserved, asof_value, now))
        upserted += 1
    
    conn.commit()
    conn.close()
    
    return {"success": True, "upserted": upserted, "branch_id": branch_id, "apt_id": apt_id}


@router.get("/branch/{branch_id}/classes")
def get_branch_classes(branch_id: str, apt_id: str, date: Optional[str] = None):
    """Get cached classes for a branch."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if date:
        # Filter by date (classes where start contains the date string)
        cursor.execute("""
            SELECT * FROM branch_class_cache 
            WHERE branch_id = ? AND apt_id = ? AND start LIKE ?
            ORDER BY start
        """, (branch_id, apt_id, f"{date}%"))
    else:
        cursor.execute("""
            SELECT * FROM branch_class_cache 
            WHERE branch_id = ? AND apt_id = ?
            ORDER BY start
        """, (branch_id, apt_id))
    
    classes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"branch_id": branch_id, "apt_id": apt_id, "classes": classes, "count": len(classes)}


@router.get("/classes")
def list_all_classes(apt_id: Optional[str] = None):
    """List all cached classes, optionally filtered by apt_id."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if apt_id:
        cursor.execute("SELECT * FROM branch_class_cache WHERE apt_id = ? ORDER BY start", (apt_id,))
    else:
        cursor.execute("SELECT * FROM branch_class_cache ORDER BY start")
    
    classes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"classes": classes, "count": len(classes)}
