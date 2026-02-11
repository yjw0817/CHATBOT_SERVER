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


@router.delete("/apartments/{apt_id}")
def delete_apartment(apt_id: str):
    """Delete apartment and all related data (cascading)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT apt_id FROM apartments WHERE apt_id = ?", (apt_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Apartment not found")

    cursor.execute("SELECT doc_id FROM documents WHERE apt_id = ?", (apt_id,))
    doc_ids = [r["doc_id"] for r in cursor.fetchall()]
    cursor.execute("SELECT conversation_id FROM conversations WHERE apt_id = ?", (apt_id,))
    conv_ids = [r["conversation_id"] for r in cursor.fetchall()]
    deleted = {}
    if doc_ids:
        ph = ",".join("?" * len(doc_ids))
        for tbl in ["manual_section_revisions", "manual_sections", "qa_issues", "chunks", "api_specs"]:
            cursor.execute(f"DELETE FROM {tbl} WHERE doc_id IN ({ph})", doc_ids)
            deleted[tbl] = cursor.rowcount
    if conv_ids:
        ph2 = ",".join("?" * len(conv_ids))
        cursor.execute(f"DELETE FROM messages WHERE conversation_id IN ({ph2})", conv_ids)
        deleted["messages"] = cursor.rowcount
    for tbl in ["conversations", "improve_suggestions", "branch_class_cache", "documents"]:
        cursor.execute(f"DELETE FROM {tbl} WHERE apt_id = ?", (apt_id,))
        deleted[tbl] = cursor.rowcount
    cursor.execute("DELETE FROM apartments WHERE apt_id = ?", (apt_id,))
    deleted["apartments"] = cursor.rowcount
    conn.commit()
    conn.close()
    return {"success": True, "apt_id": apt_id, "deleted": deleted}


@router.delete("/doc/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all related data."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    deleted = {}
    for tbl in ["manual_section_revisions", "manual_sections", "qa_issues", "chunks", "api_specs"]:
        cursor.execute(f"DELETE FROM {tbl} WHERE doc_id = ?", (doc_id,))
        deleted[tbl] = cursor.rowcount
    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    deleted["documents"] = cursor.rowcount
    conn.commit()
    conn.close()
    return {"success": True, "doc_id": doc_id, "deleted": deleted}


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

MANUALIZE_PROMPT = """당신은 영업/운영 문서를 **RAG(검색 기반 답변)**에 넣기 적합한 **구조화 매뉴얼 데이터**로 변환하는 전문가입니다.

[목표]
- 원문(raw_text)의 **사실/구조(헤딩/번호/목차/구분)**를 **있는 그대로 보존**하면서, 검색/인용/검증/업데이트가 쉬운 **정제된 매뉴얼 JSON**을 만듭니다.
- 원문에 없는 정보를 **추가/추측/창작하지 않습니다.**
- 개인정보(PII)는 **탐지 + 마스킹**하여 RAG에 안전하게 저장 가능하게 만듭니다.

[입력]
raw_text: {raw_text}

[출력 형식]
- 아래 스키마를 만족하는 **RFC8259 유효 JSON**만 반환하십시오.
- 코드블록(```), 주석, 설명 문장 금지. **오직 JSON 텍스트만** 출력합니다.
- 모든 문자열은 큰따옴표(")를 사용하세요. trailing comma 금지.
- 아래 스키마의 모든 필드를 **반드시 포함**하세요. (해당 없음이면 빈 문자열 "" 또는 빈 배열 [] 사용)

{{
  "doc_title": "문서 제목(원문에서 추출, 없으면 빈 문자열)",
  "doc_type": "POLICY|PROCESS|FAQ|NOTICE|MIXED",
  "summary": "문서 핵심 2~4문장 요약(추측 금지, 줄바꿈 없이)",
  "sections": [
    {{
      "section_id": "stable_slug_like_this",
      "name": "섹션 이름(원문 헤딩/번호 제목 그대로)",
      "tags": ["키워드", "업무영역", "대상", "아파트명/지점명(있으면)"],
      "content": [
        {{
          "rule_id": "S1-R1",
          "title": "항목 제목(짧게, 원문 소제목/문단 주제 기반)",
          "bullets": [
            "원문에서 확인되는 규칙/정의/절차/안내를 짧은 bullet로 정리(추측 금지)"
          ],
          "structured": {{
            "target": "대상(원문에 명시된 경우만, 없으면 빈 문자열)",
            "condition": "적용 조건(원문에 명시된 경우만, 없으면 빈 문자열)",
            "procedure": [],
            "exceptions": [],
            "owner": "담당/주체(원문에 명시된 경우만, 없으면 빈 문자열)",
            "channel": "문의/접수 채널(원문에 명시된 경우만, 없으면 빈 문자열)"
          }},
          "source_quotes": [],
          "issues": []
        }}
      ]
    }}
  ],
  "clarification_questions": [],
  "pii_handling": {{
    "pii_found": false,
    "pii_types": [],
    "masking_policy": []
  }},
  "change_summary": "이번 변환에서 수행한 작업 요약(사실 추가 금지)"
}}

[문서 타입(doc_type) 판정 규칙]
- POLICY: 해야/금지/조건/기준/규정 중심
- PROCESS: 단계/절차/흐름(업무 프로세스) 중심
- FAQ: 질문-답변(Q/A) 다수
- NOTICE: 공지/안내/소개/변경사항 중심
- MIXED: 위 성격이 혼합되어 지배적인 하나로 단정하기 어려움

[섹션 구성 규칙 - 최우선(중요)]
1) `sections`는 **원문에 존재하는 헤딩/번호/목차/구분 구조를 그대로** 매핑합니다.
2) 임의로 섹션을 **늘리거나/줄이거나/병합하거나/분리하지 마세요.**
3) 원문에 섹션 구분이 전혀 없다면 `sections`는 **1개만** 만들고:
   - section_id="general"
   - name="일반"
4) `name`은 원문 섹션 제목을 가능한 그대로 사용합니다(번호 포함 가능).
5) `section_id`는 `name` 기반 안정 slug(영문 소문자 + 하이픈). 섹션명이 없으면 "general".
6) `rule_id`는 섹션 순서/항목 순서 기준으로 고정:
   - 첫 섹션의 첫 항목: "S1-R1"
   - 첫 섹션의 둘째 항목: "S1-R2"
   - 둘째 섹션의 첫 항목: "S2-R1" ... 방식

[항목(rule) 추출 규칙]
- 각 섹션 안에서 원문에 존재하는 소제목(###), 번호 목록, 불릿 목록, 문단 주제 단위로 rule을 구성합니다.
- 원문에 불릿 목록이 있으면 bullets에 **항목을 그대로(의미 보존)** 나열합니다. (문장 다듬기 최소화)
- 원문에 "버전/웹사이트/연락처" 같은 메타 정보가 있으면 각각 별도 rule로 분리(단, 원문 구조가 이미 분리돼 있지 않다면 문단 단위에서만 분리).
- 서로 다른 위치에 흩어진 동일 주제 정보라도 **원문이 동일 섹션 안**에 있고 명확히 같은 주제일 때만, 한 rule로 합칠 수 있습니다.
  (다른 섹션의 내용을 끌어와 합치지 마세요)

[정제 규칙]
- 원문 표현을 과도하게 미화/확장하지 말고, **짧고 명확한 사실/규칙 형태**로 정리합니다.
- 원문에 없는 "대상/담당/채널/절차"를 **일반 상식으로 채우지 마세요.**
- `structured`는 원문에 명시된 것만 채우고, 없으면:
  - 문자열 필드: ""
  - 배열 필드(procedure/exceptions): []

[source_quotes 규칙(근거 인용)]
- 각 rule마다 원문 근거를 0~2개까지 `source_quotes`에 넣습니다.
- 각 quote는 원문에서 **연속된 문자열**을 그대로 가져오되, 길이는:
  - 영어: 최대 25단어
  - 한국어: 50자 내외(가능하면 20~70자 범위)
- 근거가 명확하지 않으면 `source_quotes`는 빈 배열 []로 둡니다.
- PII가 포함된 quote는 아래 규칙대로 마스킹 후 인용합니다.

[이슈 탐지 규칙]
- issues는 문제가 없으면 반드시 [] 입니다.
- MISSING: 필수 정보 누락(기한/금액/담당/채널/조건 등)
- AMBIGUOUS: 해석이 갈리는 표현("적당히", "빠르게", "가능하면" 등) 또는 정의 불명
- CONFLICT: 문서 내 상충 규칙/예외 충돌
- PII_RISK: 개인정보/식별정보 포함 또는 포함 가능성 높음(마스킹 필요 포함)
- API_NEEDED: 자동화/조회가 필요하지만 API/권한/데이터가 명시되지 않음

[PII 처리 규칙(중요)]
- 원문에 PII가 있으면:
  1) bullets 및 source_quotes에는 **마스킹된 형태로만** 남깁니다.
  2) 해당 rule의 issues에 PII_RISK를 기록하고 severity는 MEDIUM 이상.
  3) pii_handling.pii_found=true
  4) pii_types에 탐지된 유형을 추가(예: ["PHONE","EMAIL"])
  5) masking_policy에는 이번 문서에서 실제 적용한 정책만 기록(예: ["PHONE: 010-****-1234","EMAIL: ab***@domain.com"])
- 원문에 PII가 없으면:
  - pii_handling.pii_found=false, pii_types=[], masking_policy=[]

[마스킹 포맷]
- 전화번호: 010-****-1234
- 이메일: ab***@domain.com
- 주소: 시/구까지만 남기고 상세는 ***
- 계좌/카드/식별번호: 뒤 4~7자리 **** 처리
- 실명: 필요 시 일부만 남기고 *** 처리

[clarification_questions 규칙]
- 확인이 필요한 질문만 0~10개.
- 원문에 없는 핵심값(금액/기한/담당/채널/예외/정의)이 실제 운영 판단에 필요한 경우에만 작성.
- 사소한 문장 다듬기 질문은 금지.

[금지]
- 원문에 없는 정보 생성/추측 금지
- 원문 사실과 다른 내용 생성 금지
- 추측/일반 상식으로 빈칸 채우기 금지
- JSON 외 텍스트 출력 금지"""


@router.post("/doc/{doc_id}/manualize")
def manualize(doc_id: str, force: bool = False):
    """Convert raw text to structured manual sections using V2 RAG-optimized prompt."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = doc["raw_text"]
    if not raw_text or raw_text.strip() == "" or raw_text.startswith("["):
        conn.close()
        error_msg = "추출된 텍스트가 없거나 유효하지 않습니다. 먼저 'Extract'를 수행해 주세요."
        if raw_text and raw_text.startswith("["):
            error_msg = f"텍스트 추출에 문제가 있습니다: {raw_text}"
        raise HTTPException(status_code=400, detail=error_msg)

    # Check if manual sections already exist (return cached if not forced)
    if not force:
        cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = ?", (doc_id,))
        existing = cursor.fetchall()
        if existing:
            sections = {row["section_name"]: row["section_text"] for row in existing}
            conn.close()
            return {
                "success": True,
                "doc_id": doc_id,
                "sections": list(sections.keys()),
                "section_details": sections,
                "llm_used": False,
                "cached": True,
                "todo_questions": []
            }

    # LLM is required
    if not is_llm_available():
        conn.close()
        raise HTTPException(status_code=503, detail="LLM이 활성화되지 않았습니다. .env 설정을 확인하세요.")

    result_data = {}
    try:
        content = call_llm(MANUALIZE_PROMPT.format(raw_text=raw_text[:8000]), temperature=0.3)
        if content:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result_data = json.loads(json_match.group())
        if not result_data.get("sections"):
            conn.close()
            raise HTTPException(status_code=502, detail="LLM 응답을 파싱할 수 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=502, detail=f"LLM 호출 실패: {str(e)}")

    sections_list = result_data.get("sections", [])
    todo_questions = result_data.get("clarification_questions", [])

    sections_map = {}
    all_issues = []

    if sections_list:
        for s in sections_list:
            name = s.get("name", "미분류")
            content_items = s.get("content", "")
            
            # V2: content is array of rule objects → flatten to readable text
            if isinstance(content_items, list):
                text_parts = []
                for rule in content_items:
                    title = rule.get("title", "")
                    if title:
                        text_parts.append(f"### {title}")
                    
                    # Bullets
                    for bullet in rule.get("bullets", []):
                        text_parts.append(f"- {bullet}")
                    
                    # Structured info
                    structured = rule.get("structured", {})
                    if structured:
                        details = []
                        if structured.get("target"):
                            details.append(f"- 대상: {structured['target']}")
                        if structured.get("condition"):
                            details.append(f"- 조건: {structured['condition']}")
                        if structured.get("procedure"):
                            for i, step in enumerate(structured["procedure"], 1):
                                details.append(f"  {i}. {step}")
                        if structured.get("exceptions"):
                            for exc in structured["exceptions"]:
                                details.append(f"- ⚠️ 예외: {exc}")
                        if structured.get("owner"):
                            details.append(f"- 담당: {structured['owner']}")
                        if structured.get("channel"):
                            details.append(f"- 채널: {structured['channel']}")
                        if details:
                            text_parts.extend(details)
                    
                    # Source quotes
                    quotes = rule.get("source_quotes", [])
                    if quotes:
                        text_parts.append(f"  📌 근거: {'; '.join(quotes)}")
                    
                    text_parts.append("")  # blank line between rules
                    
                    # Collect issues from each rule
                    for issue in rule.get("issues", []):
                        severity_map = {"HIGH": "RED", "MEDIUM": "YELLOW", "LOW": "YELLOW"}
                        all_issues.append({
                            "severity": severity_map.get(issue.get("severity", "MEDIUM"), "YELLOW"),
                            "issue_type": issue.get("type"),
                            "message": f"[{name}] {issue.get('message', '')}",
                            "suggestion": issue.get("suggestion", "")
                        })
                
                sections_map[name] = "\n".join(text_parts).strip()
            else:
                # V1 fallback: content is plain string
                sections_map[name] = content_items if content_items else "정보 없음"

                # V1 issues at section level
                for issue in s.get("issues", []):
                    all_issues.append({
                        "severity": "RED" if issue.get("type") in ("MISSING", "CONFLICT", "PII_RISK") else "YELLOW",
                        "issue_type": issue.get("type"),
                        "message": f"[{name}] {issue.get('message')}",
                        "suggestion": issue.get("suggestion")
                    })
    
    # Save sections
    cursor.execute("DELETE FROM manual_sections WHERE doc_id = ?", (doc_id,))
    for section_name, section_text in sections_map.items():
        section_id = f"sec_{uuid.uuid4().hex[:8]}"
        cursor.execute(
            "INSERT INTO manual_sections (section_id, doc_id, section_name, section_text) VALUES (?, ?, ?, ?)",
            (section_id, doc_id, section_name, section_text if section_text else "정보 없음")
        )
    
    # Save issues found during manualization
    cursor.execute("DELETE FROM qa_issues WHERE doc_id = ?", (doc_id,))
    for issue in all_issues:
        issue_id = f"issue_{uuid.uuid4().hex[:8]}"
        cursor.execute(
            "INSERT INTO qa_issues (issue_id, doc_id, severity, issue_type, message, suggestion, status) VALUES (?, ?, ?, ?, ?, ?, 'OPEN')",
            (issue_id, doc_id, issue["severity"], issue["issue_type"], issue["message"], issue["suggestion"])
        )
    
    cursor.execute("UPDATE documents SET updated_at = ? WHERE doc_id = ?", (datetime.now().isoformat(), doc_id))
    conn.commit()

    # Auto Gate#1: run gate check on each section after manualize
    gate_results = {}
    if is_llm_available():
        raw_text_for_gate = raw_text[:4000] if raw_text else ""
        for section_name, section_text in sections_map.items():
            try:
                gate_content = call_llm(GATE_CHECK_PROMPT.format(
                    section_text=section_text[:3000],
                    raw_text=raw_text_for_gate
                ), temperature=0.3)
                gate_data = {"status": "PASS", "score": 100, "reasons": [], "required_actions": []}
                if gate_content:
                    gm = re.search(r'\{[\s\S]*\}', gate_content)
                    if gm:
                        gate_data = json.loads(gm.group())
                gate_results[section_name] = gate_data

                # Save gate result
                cursor.execute(
                    "UPDATE manual_sections SET gate_status = ?, gate_score = ?, gate_reasons_json = ?, gate_stale = 0, updated_at = ? WHERE doc_id = ? AND section_name = ?",
                    (gate_data.get("status", "PASS"), gate_data.get("score", 100),
                     json.dumps(gate_data.get("reasons", []), ensure_ascii=False),
                     datetime.now().isoformat(), doc_id, section_name)
                )
            except Exception as e:
                print(f"[MANUALIZE_GATE] Gate error for '{section_name}': {e}")
                gate_results[section_name] = {"status": "PASS", "score": 100, "reasons": []}
        conn.commit()

    conn.close()

    return {
        "success": True,
        "doc_id": doc_id,
        "sections": list(sections_map.keys()),
        "section_details": sections_map,
        "todo_questions": todo_questions,
        "change_summary": result_data.get("change_summary", ""),
        "pii_handling": result_data.get("pii_handling", {}),
        "gate_results": gate_results,
        "llm_used": True
    }


# ============ STEP 2: SECTION GATE (per-section AI check) ============

GATE_CHECK_PROMPT = """당신은 RAG 반영 전, 매뉴얼 섹션 텍스트(section_text)의 품질/리스크를 판정하는 QA 게이트입니다.
중요: 내용을 새로 작성하거나 고치지 말고, 오직 '검증 결과'만 출력하세요.

[입력]
section_text: {section_text}
raw_text: {raw_text}

[판정 상태]
- PASS: 바로 RAG 반영 가능
- NEED_FIX: 사람이 수정/보강 후 반영 권장
- BLOCK: RAG 반영 금지(보안/심각 충돌/형식 붕괴)

[검증 항목]
1) PII_RISK (BLOCK 우선)
- 전화번호/이메일/계좌/상세주소/식별번호가 마스킹 없이 노출되면 BLOCK
- 마스킹 규칙 예: 010-****-1234, ab***@domain.com, 상세주소 ***, 계좌번호 ****

2) CONFLICT (BLOCK 또는 NEED_FIX)
- 같은 문서/섹션 내 상충 규칙(환불 가능 vs 불가 등) 징후
- 서로 다른 조건이 충돌하는데 우선순위/예외가 명시되지 않음

3) MISSING/AMBIGUOUS (NEED_FIX)
- 필수값(기한/금액/담당/채널/조건) 누락
- 모호 표현(적당히/가능하면/상황에 따라/신속히 등) 과다

4) HALLUCINATION_RISK (NEED_FIX 또는 BLOCK)
- raw_text에 근거가 보이지 않는 구체 수치/기간/금액/정책이 섞여 들어간 흔적
- 특히 "반드시", "무조건", "항상"처럼 단정적 표현이 근거 없이 추가된 경우

5) FORMAT (NEED_FIX)
- "## 섹션" / "### 항목" / "-" bullet 형식 붕괴
- 섹션명/항목명이 의미 없이 비어 있거나 반복됨

[출력 형식]
- 아래 RFC8259 유효 JSON만 반환하세요. (코드블록/설명 금지)
- score는 0~100 정수

{{
  "status": "PASS|NEED_FIX|BLOCK",
  "score": 0,
  "reasons": [
    {{
      "type": "PII_RISK|CONFLICT|MISSING|AMBIGUOUS|HALLUCINATION_RISK|FORMAT",
      "severity": "LOW|MEDIUM|HIGH",
      "message": "무엇이 문제인지",
      "location_hint": "가능하면 섹션/항목 제목 또는 문제 라인 일부",
      "fix_suggestion": "어떻게 고치면 되는지(짧게)"
    }}
  ],
  "required_actions": [
    "사용자가 해야 할 조치 1",
    "조치 2"
  ]
}}

[판정 가이드]
- BLOCK: (1) PII 미마스킹 노출, (2) 심각한 충돌, (3) 근거 없는 수치/금액/기한 다수, (4) 형식 붕괴 심각
- NEED_FIX: 모호/누락이 핵심 답변에 영향을 주는 수준, 또는 [확인 필요]가 과도하거나 핵심값이 비어 있음
- PASS: 위 문제 없음 또는 경미(LOW)이며 답변 품질에 영향 적음
"""


@router.post("/doc/{doc_id}/section/{section_name}/gate")
def gate_section(doc_id: str, section_name: str):
    """Run AI gate check on a single section."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT section_id, section_text FROM manual_sections WHERE doc_id = ? AND section_name = ?",
                   (doc_id, section_name))
    sec = cursor.fetchone()
    if not sec:
        conn.close()
        raise HTTPException(status_code=404, detail="Section not found")

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    raw_text = (doc["raw_text"] or "")[:4000] if doc else ""

    gate_result = {"status": "PASS", "score": 100, "reasons": [], "required_actions": []}

    if is_llm_available():
        try:
            content = call_llm(GATE_CHECK_PROMPT.format(
                section_text=sec["section_text"][:3000],
                raw_text=raw_text
            ), temperature=0.3)
            if content:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    gate_result = json.loads(json_match.group())
        except Exception as e:
            print(f"[GATE_SECTION] LLM error: {e}")

    # Save gate result to manual_sections
    cursor.execute(
        "UPDATE manual_sections SET gate_status = ?, gate_score = ?, gate_reasons_json = ?, gate_stale = 0, updated_at = ? WHERE section_id = ?",
        (gate_result.get("status", "PASS"), gate_result.get("score", 100),
         json.dumps(gate_result.get("reasons", []), ensure_ascii=False),
         datetime.now().isoformat(), sec["section_id"])
    )
    conn.commit()
    conn.close()

    return {"success": True, "section_name": section_name, **gate_result}


@router.post("/doc/{doc_id}/section/{section_name}/gate-stale")
def set_gate_stale(doc_id: str, section_name: str):
    """Mark section as gate_stale (saved without gate re-check)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE manual_sections SET gate_stale = 1, updated_at = ? WHERE doc_id = ? AND section_name = ?",
        (datetime.now().isoformat(), doc_id, section_name)
    )
    conn.commit()
    conn.close()
    return {"success": True}


# ============ STEP 2: QUALITY GATE (document-level) ============

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


# ============ STEP 2: AI HELPER (Fill/Refine) ============

class RefineRequest(BaseModel):
    text: str
    task: str  # "refine", "fill", "recommend"
    context: Optional[str] = None  # Section name or issue message
    allow_qa: Optional[str] = None  # "true" or "false" (for fill task)


def _to_bool_allow_qa(val) -> bool:
    """Normalize allow_qa to bool. Accepts bool, str, int, None."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    if isinstance(val, (int, float)):
        return bool(val)
    return False


FILL_SECTION_TEXT_PROMPT_V3 = """당신은 RAG 청크(섹션 텍스트)를 '독립적으로 이해 가능한 매뉴얼'로 보강하는 편집자입니다.
중요: 이 작업은 '새 정보 추가'가 아니라, 동일 문서(raw_text) 내부의 관련 내용을 모아 재구성하는 것입니다.

[입력]
section_text: {section_text}
raw_text: {raw_text}

[절대 규칙]
1) 원문에 없는 정보(수치/기간/금액/정책/예외)를 절대 만들지 마세요.
2) 근거가 없으면 내용을 채우지 말고, 해당 지점에만 "[확인 필요: 무엇을 확인?]" 라벨을 붙이세요.
3) 암묵 조건/전제는 raw_text에 암시/표현이 있는 경우에만 명시적으로 풀어쓰세요.
4) 약어/내부 용어는 원문에 등장한 것만 풀어 설명을 추가하세요. (원문에 없으면 금지)
5) 개인정보(전화/이메일/계좌/상세주소/식별번호 등)는 ***로 마스킹을 유지하세요. 원문에 있어도 그대로 노출 금지.
6) [Q&A 정책] {qa_policy_text}
7) 기존 section_text의 주제/범위를 바꾸지 마세요. (다른 섹션 주제를 섞어 넣지 말 것)

[개선 목표]
- 앞뒤 섹션 없이도 이 section_text만 읽고 답변 가능한 수준으로,
  원문에 흩어진 관련 규칙/조건/채널/예외를 이 섹션 안에 통합하세요.
- 중복 bullet 제거, 표현 정돈, 항목 제목을 명확히.
- 너무 긴 항목은 같은 주제 안에서 2개로 쪼개되 형식은 유지하세요.

[출력 형식(반드시 유지)]
- 오직 개선된 section_text 전체를 plain text로만 출력하세요.
- 섹션 시작: "## "
- 항목 제목: "### "
- 본문: "-" bullet
- Q&A는 allow_qa=true 이거나 원래 존재하는 경우에만, 항목 하단에 아래 형식으로만 포함:
  - Q: ...
  - A: ...

[추가 가이드]
- raw_text에서 근거가 명확한 내용만 '모아서' 넣고, 근거 없는 부분은 채우지 않습니다.
- "[확인 필요]"는 남발하지 말고 '필수 판단에 필요한 핵심 빈칸'에만 사용하세요.
- 설명/해설/JSON 출력 금지. 오직 최종 텍스트만 출력하세요.
"""


FINALIZE_SECTION_TEXT_PROMPT_V1 = """당신은 RAG 검색 적중률과 답변 일관성을 높이기 위해 section_text(plain text)를 '최종 문구'로 다듬는 편집자입니다.
중요: 사실/정책/수치/기간/금액 등 새로운 내용을 추가하지 마세요. 오직 표현과 구조만 최적화합니다.

[입력]
section_text: {section_text}
raw_text: {raw_text}

[절대 규칙]
1) 새 정보 추가 금지(수치/기간/금액/정책/예외/절차 창작 금지)
2) 원문/현 section_text와 다른 사실 생성 금지
3) 개인정보 마스킹 유지(***)
4) 근거가 불명확한 문장 추가 금지. 불명확하면 "[확인 필요: ...]"를 유지하거나 더 명확히 작성

[최적화 목표]
- 검색 키워드에 잘 걸리도록 항목 제목(###)을 '질문형 또는 키워드형'으로 선명하게
  예: "환불 규정" → "환불 규정/기한/위약금"
- bullets를 짧고 병렬 구조로 정리(중복 제거)
- 같은 의미의 표현을 통일(용어 표준화)
- Q&A가 이미 존재하면, 질문을 더 명확히 하되 답은 그대로(내용 추가 금지)
- 너무 긴 bullet은 2개로 분리하되 의미 유지

[출력]
- 오직 최종 section_text 전체를 plain text로만 출력하세요.
- 형식 유지:
  - "## " 섹션
  - "### " 항목
  - "-" bullet
  - Q/A는 존재할 때만 유지
- 설명/해설/JSON 금지
"""


@router.post("/doc/{doc_id}/refine-text")
def refine_text(doc_id: str, req: RefineRequest):
    """AI helper: fill (맥락 보강), refine (RAG 최적화), recommend (표준 템플릿 제안)."""
    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM 사용 불가")

    # Fetch original document raw_text for reference
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    conn.close()

    raw_text = ""
    if doc and doc["raw_text"]:
        raw_text = doc["raw_text"][:4000]

    raw_text_safe = raw_text or "(원본 문서를 찾을 수 없습니다. 기존 텍스트만 참고하세요.)"

    try:
        if req.task == "fill":
            allow_qa = _to_bool_allow_qa(req.allow_qa)
            qa_policy_text = (
                "원문 근거가 명확한 경우에만 Q&A를 1~3개 추가할 수 있습니다. 답이 불명확하면 Q&A를 만들지 말고 [확인 필요]로 처리하세요."
                if allow_qa else
                "Q&A는 새로 추가하지 마세요. 기존 Q&A만 유지/정리하세요."
            )
            prompt = FILL_SECTION_TEXT_PROMPT_V3.format(
                section_text=req.text,
                raw_text=raw_text_safe,
                qa_policy_text=qa_policy_text
            )
        elif req.task == "refine":
            prompt = FINALIZE_SECTION_TEXT_PROMPT_V1.format(
                section_text=req.text,
                raw_text=raw_text_safe
            )
        else:
            # recommend: use fill prompt with Q&A disabled
            prompt = FILL_SECTION_TEXT_PROMPT_V3.format(
                section_text=req.text,
                raw_text=raw_text_safe,
                qa_policy_text="Q&A는 새로 추가하지 마세요. 기존 Q&A만 유지/정리하세요."
            )

        suggestion = call_llm(prompt, temperature=0.3)
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
    cursor.execute("SELECT section_name, section_text, gate_status, gate_reasons_json, gate_stale FROM manual_sections WHERE doc_id = ?", (doc_id,))
    sections = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sections


def _split_raw_by_headings(raw_text: str) -> list:
    """Split raw_text into chunks by heading patterns."""
    heading_pattern = re.compile(r'^(?:#{1,4}\s+.+|(?:\d+[\.\)]\s*).+|.+[:\uff1a]\s*)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(raw_text))
    if not matches:
        return []
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        heading = m.group().strip().lstrip("#").strip()
        body = raw_text[start:end].strip()
        chunks.append({"heading": heading, "body": body})
    return chunks


def _heading_match(section_name: str, headings: list) -> int:
    """Match section name to heading index by word overlap. Returns -1 if no match."""
    sec_words = set(re.findall(r'[\w가-힣]+', section_name.lower()))
    if not sec_words:
        return -1
    best_idx, best_score = -1, 0
    for i, h in enumerate(headings):
        h_words = set(re.findall(r'[\w가-힣]+', h["heading"].lower()))
        overlap = len(sec_words & h_words)
        score = overlap / max(len(sec_words | h_words), 1)
        if score > best_score and score >= 0.3:
            best_score = score
            best_idx = i
    return best_idx


@router.get("/doc/{doc_id}/source-map")
def get_source_map(doc_id: str):
    """Get per-section matched raw text using heading matching."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    cursor.execute("SELECT section_name FROM manual_sections WHERE doc_id = ?", (doc_id,))
    sections = [row["section_name"] for row in cursor.fetchall()]
    conn.close()

    raw_text = doc["raw_text"] or ""
    chunks = _split_raw_by_headings(raw_text)
    if not chunks:
        source_map = {name: None for name in sections}
        return {"source_map": source_map, "raw_text": raw_text, "matched": 0}

    # 1) Find matched heading index for each section (in order)
    match_indices = []
    for sec_name in sections:
        idx = _heading_match(sec_name, chunks)
        match_indices.append(idx)

    # 2) For each section, extract raw_text from matched heading to next matched heading
    source_map = {}
    for i, sec_name in enumerate(sections):
        idx = match_indices[i]
        if idx < 0:
            source_map[sec_name] = None
            continue
        # Find the next section's matched heading index that comes after this one
        next_chunk_idx = len(chunks)
        for j in range(i + 1, len(sections)):
            if match_indices[j] > idx:
                next_chunk_idx = match_indices[j]
                break
        # Collect raw_text from matched heading to next boundary
        start_pos = chunks[idx]["body"]  # not useful, use raw positions
        # Rebuild from chunk bodies between idx and next_chunk_idx
        parts = [chunks[k]["body"] for k in range(idx, min(next_chunk_idx, len(chunks)))]
        source_map[sec_name] = "\n\n".join(parts)

    return {"source_map": source_map, "raw_text": raw_text, "matched": sum(1 for v in source_map.values() if v is not None)}


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
