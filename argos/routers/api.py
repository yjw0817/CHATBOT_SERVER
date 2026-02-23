"""API routes for apartments, documents, and file upload."""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from database import get_connection

router = APIRouter(prefix="/api", tags=["api"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Import unified LLM client
from llm_client import call_llm, is_llm_available, get_llm_info, LLM_MODEL, set_llm_mode, get_llm_mode, set_llm_model, get_llm_model, ping_llm, call_vision_llm

# Import RAG index (FAISS + embedding)
import numpy as np
from rag_index import get_embedding, build_index as rag_build_index, search as faiss_search, invalidate_index

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


@router.get("/llm/mode")
def get_mode():
    """Get current LLM mode (local/remote)."""
    return {"mode": get_llm_mode(), "info": get_llm_info()}


def _fetch_model_list() -> list[dict]:
    """Fetch available models from current Ollama endpoint."""
    import requests as _req
    info = get_llm_info()
    base_url = (info.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
    headers = {}
    api_key = os.getenv("LLM_API_KEY", "")
    if info.get("mode") == "remote" and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = _req.get(f"{base_url}/api/tags", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size_gb = round(m.get("size", 0) / 1e9, 1) if m.get("size") else None
            models.append({"name": name, "size_gb": size_gb})
        models.sort(key=lambda x: x["name"])
        return models
    except Exception:
        return []


@router.post("/llm/mode")
def switch_mode(mode: str = Form(...)):
    """Switch LLM mode at runtime. Returns model list for the new mode."""
    try:
        set_llm_mode(mode)
        models = _fetch_model_list()
        return {"success": True, "mode": get_llm_mode(), "info": get_llm_info(), "models": models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/llm/model")
def get_model():
    """Get current LLM model."""
    return {"model": get_llm_model(), "info": get_llm_info()}


@router.post("/llm/model")
def switch_model(model: str = Form(...)):
    """Switch LLM model at runtime."""
    set_llm_model(model)
    return {"success": True, "model": get_llm_model(), "info": get_llm_info()}


@router.get("/llm/models")
def list_models():
    """List available models from current Ollama endpoint (local or remote)."""
    info = get_llm_info()
    models = _fetch_model_list()
    return {"models": models, "mode": info.get("mode"), "current": get_llm_model()}


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
            "INSERT INTO apartments (apt_id, name) VALUES (%s, %s)",
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
    cursor.execute("SELECT apt_id FROM apartments WHERE apt_id = %s", (apt_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Apartment not found")

    cursor.execute("SELECT doc_id FROM documents WHERE apt_id = %s", (apt_id,))
    doc_ids = [r["doc_id"] for r in cursor.fetchall()]
    cursor.execute("SELECT conversation_id FROM conversations WHERE apt_id = %s", (apt_id,))
    conv_ids = [r["conversation_id"] for r in cursor.fetchall()]
    deleted = {}
    if doc_ids:
        ph = ",".join(["%s"] * len(doc_ids))
        for tbl in ["manual_section_revisions", "manual_sections", "qa_issues", "chunks", "api_specs", "doc_chunks"]:
            cursor.execute(f"DELETE FROM {tbl} WHERE doc_id IN ({ph})", doc_ids)
            deleted[tbl] = cursor.rowcount
    if conv_ids:
        ph2 = ",".join(["%s"] * len(conv_ids))
        cursor.execute(f"DELETE FROM messages WHERE conversation_id IN ({ph2})", conv_ids)
        deleted["messages"] = cursor.rowcount
    for tbl in ["conversations", "improve_suggestions", "branch_class_cache", "documents"]:
        cursor.execute(f"DELETE FROM {tbl} WHERE apt_id = %s", (apt_id,))
        deleted[tbl] = cursor.rowcount
    cursor.execute("DELETE FROM apartments WHERE apt_id = %s", (apt_id,))
    deleted["apartments"] = cursor.rowcount
    conn.commit()
    conn.close()
    return {"success": True, "apt_id": apt_id, "deleted": deleted}


@router.delete("/doc/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all related data."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM documents WHERE doc_id = %s", (doc_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    deleted = {}
    for tbl in ["manual_section_revisions", "manual_sections", "qa_issues", "chunks", "api_specs", "doc_chunks"]:
        cursor.execute(f"DELETE FROM {tbl} WHERE doc_id = %s", (doc_id,))
        deleted[tbl] = cursor.rowcount
    cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
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
        "SELECT doc_id, version FROM documents WHERE apt_id = %s AND content_hash = %s AND status != 'ARCHIVED' ORDER BY version DESC LIMIT 1",
        (apt_id, content_hash)
    )
    existing = cursor.fetchone()
    
    if existing:
        new_version = existing["version"] + 1
        cursor.execute(
            "UPDATE documents SET status = 'ARCHIVED', updated_at = %s WHERE apt_id = %s AND content_hash = %s AND status != 'ARCHIVED'",
            (datetime.now().isoformat(), apt_id, content_hash)
        )
    else:
        cursor.execute("SELECT MAX(version) as max_ver FROM documents WHERE apt_id = %s", (apt_id,))
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
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'DRAFT', %s, %s)
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
            "SELECT doc_id, apt_id, title, source_filename, source_type, version, status, created_at FROM documents WHERE apt_id = %s ORDER BY created_at DESC",
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

VISION_PROMPT = "이 이미지를 한국어로 자세히 설명해줘. UI 화면이라면 어떤 기능의 화면인지, 버튼/메뉴/입력 필드 등 구성 요소를 포함해서 설명해."
VISION_OCR_PROMPT = "이 이미지에 있는 텍스트를 한국어로 정확히 읽어서 그대로 출력해줘. 목차라면 항목과 페이지 번호를 유지해줘. 표라면 표 구조를 유지해줘. 가능한 원본 텍스트를 그대로 보존해서 출력해."

# Extract progress tracker (in-memory)
_extract_progress = {}  # {doc_id: {"page": 1, "total_pages": 10, "images_done": 2, "images_total": 5, "status": "..."}}


def _is_extract_cancelled(doc_id: str) -> bool:
    return _extract_progress.get(doc_id, {}).get("cancelled", False)


def _describe_image(image_base64: str, doc_id: str = None) -> str:
    """Vision LLM으로 이미지 설명 생성. 실패 시 [이미지 설명 실패] 반환."""
    if doc_id and _is_extract_cancelled(doc_id):
        return "[이미지 설명 건너뜀: 취소됨]"
    if doc_id and doc_id in _extract_progress:
        _extract_progress[doc_id]["images_done"] = _extract_progress[doc_id].get("images_done", 0) + 1
        _extract_progress[doc_id]["status"] = f"이미지 {_extract_progress[doc_id]['images_done']}/{_extract_progress[doc_id].get('images_total', '?')} Vision LLM 분석 중"
    try:
        desc = call_vision_llm(VISION_PROMPT, image_base64)
        if desc and desc.strip():
            return f"[이미지 설명: {desc.strip()}]"
    except Exception as e:
        print(f"[EXTRACT] Vision LLM failed: {e}")
    return "[이미지 설명 실패]"


@router.get("/doc/{doc_id}/extract-progress")
def get_extract_progress(doc_id: str):
    """Poll extract progress."""
    return _extract_progress.get(doc_id, {})


@router.post("/doc/{doc_id}/extract-cancel")
def cancel_extract(doc_id: str):
    """Cancel an in-progress extract operation."""
    if doc_id in _extract_progress:
        _extract_progress[doc_id]["cancelled"] = True
        return {"success": True, "detail": "취소 요청됨"}
    return {"success": False, "detail": "진행 중인 작업 없음"}


@router.post("/doc/{doc_id}/extract-text")
def extract_text(doc_id: str, resume_page: int = 0):
    """Extract text from uploaded document (DOCX, PDF, TXT, MD).
    resume_page: 0=처음부터, N=N페이지부터 이어서 (PDF only, 기존 raw_text에 append)
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT source_filename, source_type, raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    source_type = doc["source_type"]
    file_path = UPLOAD_DIR / f"{doc_id}.{source_type}"

    if not file_path.exists():
        conn.close()
        raise HTTPException(status_code=404, detail="File not found on disk")

    # 이어하기: 기존 텍스트 보존
    existing_text = (doc["raw_text"] or "") if resume_page > 0 else ""
    raw_text = ""
    image_count = 0

    if source_type == "docx":
        try:
            from docx import Document
            import zipfile
            import base64

            _extract_progress[doc_id] = {"page": 0, "total_pages": 0, "images_done": 0, "images_total": 0, "status": "DOCX 텍스트 추출 중", "pages": {}}

            docx_doc = Document(str(file_path))
            parts = []

            # 단락
            para_texts = []
            for p in docx_doc.paragraphs:
                if p.text.strip():
                    para_texts.append(p.text)
            if para_texts:
                para_content = "\n".join(para_texts)
                parts.append(para_content)
                _extract_progress[doc_id]["pages"]["본문"] = para_content

            # 테이블
            _extract_progress[doc_id]["status"] = "테이블 추출 중"
            for t_idx, table in enumerate(docx_doc.tables):
                rows_text = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows_text.append(" | ".join(cells))
                if rows_text:
                    table_content = "[표]\n" + "\n".join(rows_text)
                    parts.append(table_content)
                    _extract_progress[doc_id]["pages"][f"표 {t_idx + 1}"] = table_content

            # 이미지 수 미리 카운트
            with zipfile.ZipFile(str(file_path), 'r') as zf:
                media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
                _extract_progress[doc_id]["images_total"] = len(media_files)
                _extract_progress[doc_id]["status"] = f"이미지 0/{len(media_files)} Vision LLM 분석 중" if media_files else "완료 중"

                for name in media_files:
                    img_bytes = zf.read(name)
                    img_b64 = base64.b64encode(img_bytes).decode()
                    image_count += 1
                    print(f"[EXTRACT] DOCX image {image_count}: {name} ({len(img_bytes)} bytes)")
                    desc = _describe_image(img_b64, doc_id)
                    parts.append(desc)
                    _extract_progress[doc_id]["pages"][f"이미지 {image_count}"] = desc

            raw_text = "\n".join(parts)
            if _is_extract_cancelled(doc_id):
                _extract_progress[doc_id].pop("pages", None)
                _extract_progress[doc_id]["status"] = "취소됨"
            else:
                _extract_progress.pop(doc_id, None)
        except Exception as e:
            _extract_progress.pop(doc_id, None)
            conn.close()
            raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {str(e)}")

    elif source_type == "pdf":
        try:
            import fitz
            import base64
            fitz.TOOLS.mupdf_display_errors(False)

            pdf_doc = fitz.open(str(file_path))
            total_pages = len(pdf_doc)

            # 이미지 포함 페이지 수 미리 카운트
            pages_with_images = []
            for pg_idx, pg in enumerate(pdf_doc):
                if pg.get_images(full=True):
                    pages_with_images.append(pg_idx)

            # resume_page: 0-indexed internally (resume_page=3 → skip pages 0,1,2)
            start_page = max(0, resume_page - 1) if resume_page > 0 else 0
            remaining_image_pages = [p for p in pages_with_images if p >= start_page]
            _extract_progress[doc_id] = {"page": start_page, "total_pages": total_pages, "images_done": 0, "images_total": len(remaining_image_pages), "status": f"{start_page}/{total_pages} 페이지", "pages": {}}

            if start_page > 0:
                print(f"[EXTRACT] PDF resume from page {start_page + 1}/{total_pages} for {doc_id}")

            def _is_garbled(text: str) -> bool:
                """텍스트가 깨졌는지 판별. 한글이 기대되는 문서에서 한글 비율이 극히 낮으면 깨진 것."""
                import re
                if not text or len(text) < 10:
                    return False
                # 공백/줄바꿈 제거 후 실질 문자만 판별
                content = re.sub(r'\s+', '', text)
                if not content:
                    return False
                korean = len(re.findall(r'[가-힣]', content))
                alpha_num = len(re.findall(r'[a-zA-Z0-9]', content))
                readable = korean + alpha_num
                ratio = readable / len(content)
                # 한글+영숫자가 40% 미만이면 깨진 텍스트
                return ratio < 0.4

            # Phase 1: 텍스트 추출 + 렌더링 대상 결정 (순차, 빠름)
            page_texts = {}       # {page_num: text}
            image_renders = {}    # {page_num: img_b64}
            garbled_pages = set()
            for page_num, page in enumerate(pdf_doc):
                if page_num < start_page:
                    continue
                _extract_progress[doc_id]["page"] = page_num + 1
                _extract_progress[doc_id]["status"] = f"{page_num + 1}/{total_pages} 페이지 텍스트 추출"

                page_text = page.get_text().strip()
                if page_text and not _is_garbled(page_text):
                    page_texts[page_num] = page_text
                    # 텍스트 정상 + 이미지 없는 페이지는 바로 스트리밍
                    if page_num not in pages_with_images:
                        _extract_progress[doc_id]["pages"][str(page_num + 1)] = page_text
                elif page_text:
                    # 텍스트 깨짐 → Vision LLM으로 대체
                    garbled_pages.add(page_num)
                    print(f"[EXTRACT] PDF page {page_num + 1}: garbled text detected, will use Vision LLM")

                # 이미지 포함 페이지 또는 텍스트 깨진 페이지 → 렌더링
                if page_num in remaining_image_pages or page_num in garbled_pages:
                    pix = page.get_pixmap(dpi=150)
                    image_renders[page_num] = base64.b64encode(pix.tobytes("png")).decode()
                    print(f"[EXTRACT] PDF page {page_num + 1} render: {pix.width}x{pix.height}")

            _extract_progress[doc_id]["images_total"] = len(image_renders)

            pdf_doc.close()

            # Phase 2: Vision LLM 순차 호출 (Ollama 대형 모델은 동시처리 불가)
            image_descs = {}  # {page_num: desc}
            vision_total = len(image_renders)

            if image_renders:
                print(f"[EXTRACT] Vision LLM sequential: {vision_total} pages for {doc_id}")
                for vi, (pg_num, img_b64) in enumerate(sorted(image_renders.items())):
                    if _is_extract_cancelled(doc_id):
                        print(f"[EXTRACT] Cancelled at vision {vi + 1}/{vision_total}")
                        break
                    # 텍스트 깨진 페이지는 OCR 프롬프트, 이미지 포함 페이지는 설명 프롬프트
                    prompt = VISION_OCR_PROMPT if pg_num in garbled_pages else VISION_PROMPT
                    label = "OCR" if pg_num in garbled_pages else "이미지 설명"
                    _extract_progress[doc_id]["status"] = f"페이지 {pg_num + 1} {label} 중 ({vi + 1}/{vision_total})"
                    _extract_progress[doc_id]["images_done"] = vi

                    try:
                        desc = call_vision_llm(prompt, img_b64)
                        if desc and desc.strip():
                            desc = desc.strip() if pg_num in garbled_pages else f"[이미지 설명: {desc.strip()}]"
                        else:
                            desc = f"[{label} 실패]"
                    except Exception as e:
                        print(f"[EXTRACT] Vision LLM failed page {pg_num + 1}: {e}")
                        desc = f"[{label} 실패]"

                    image_descs[pg_num] = desc
                    image_count += 1
                    _extract_progress[doc_id]["images_done"] = image_count
                    # 해당 페이지 텍스트+이미지 합쳐서 스트리밍
                    combined = page_texts.get(pg_num, "")
                    combined = (combined + "\n" + desc).strip() if combined else desc
                    _extract_progress[doc_id]["pages"][str(pg_num + 1)] = combined

            # Phase 3: 페이지 순서대로 조합
            parts = []
            all_pages = sorted(set(list(page_texts.keys()) + list(image_descs.keys())))
            for pg_num in all_pages:
                page_parts = []
                if pg_num in page_texts:
                    page_parts.append(page_texts[pg_num])
                if pg_num in image_descs:
                    page_parts.append(image_descs[pg_num])
                if page_parts:
                    parts.append("\n".join(page_parts))
            raw_text = "\n".join(parts)
            if _is_extract_cancelled(doc_id):
                # 취소 시: progress 유지 (이어하기용), pages만 제거 (메모리 절약)
                _extract_progress[doc_id].pop("pages", None)
                _extract_progress[doc_id]["status"] = f"취소됨 ({_extract_progress[doc_id].get('page', 0)}/{total_pages} 페이지)"
            else:
                _extract_progress.pop(doc_id, None)
        except Exception as e:
            # 에러 시에도 progress 유지 (이어하기용)
            if doc_id in _extract_progress:
                _extract_progress[doc_id].pop("pages", None)
                _extract_progress[doc_id]["status"] = f"오류 ({_extract_progress[doc_id].get('page', 0)}/{_extract_progress[doc_id].get('total_pages', '?')} 페이지)"
            conn.close()
            raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

    elif source_type in ("txt", "md"):
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

    else:
        raw_text = f"[지원하지 않는 파일 형식: {source_type}]"

    # 이어하기: 기존 텍스트 + 새 텍스트 합침
    if existing_text and raw_text:
        raw_text = existing_text + "\n" + raw_text
    elif existing_text:
        raw_text = existing_text

    cursor.execute("UPDATE documents SET raw_text = %s, updated_at = %s WHERE doc_id = %s",
                   (raw_text, datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "doc_id": doc_id,
        "chars": len(raw_text),
        "resumed_from": resume_page if resume_page > 0 else None,
        "image_count": image_count,
        "preview": raw_text[:300] if raw_text else ""
    }


MANUALIZE_HARD_LIMIT = 30000  # chars — force window split above this (128K ctx 기준)
MANUALIZE_WINDOW_SIZE = 25000  # chars per window (~37K tokens, 128K ctx의 ~75%)
MANUALIZE_WINDOW_OVERLAP = 300

# Manualize progress tracker (in-memory, per doc_id)
_manualize_progress = {}  # {doc_id: {"done": 3, "total": 15}}


@router.get("/doc/{doc_id}/char-count")
def get_char_count(doc_id: str):
    """Get raw_text character count and window split info (lightweight pre-check)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    raw_text = row["raw_text"] or ""
    chars = len(raw_text)
    mode = get_llm_mode()
    window_count = len(_split_by_headings(raw_text))
    return {"chars": chars, "window_count": window_count, "hard_limit": MANUALIZE_HARD_LIMIT, "mode": mode}


@router.get("/doc/{doc_id}/manualize-progress")
def get_manualize_progress(doc_id: str):
    """Poll manualize progress. Returns {done, total, cancelled, sections}."""
    prog = _manualize_progress.get(doc_id, {"done": 0, "total": 0})
    return {
        "done": prog.get("done", 0),
        "total": prog.get("total", 0),
        "cancelled": prog.get("cancelled", False),
        "sections": prog.get("sections", {}),
    }


@router.post("/doc/{doc_id}/manualize-cancel")
def cancel_manualize(doc_id: str):
    """Cancel an in-progress manualize operation."""
    if doc_id in _manualize_progress:
        _manualize_progress[doc_id]["cancelled"] = True
        return {"success": True, "detail": "취소 요청됨"}
    return {"success": False, "detail": "진행 중인 작업 없음"}


# ============ STEP 2: MANUALIZE ============

MANUALIZE_PROMPT = """당신은 영업/운영 문서를 RAG(검색 기반 답변)에 넣기 위한
"구조화된 정보 추출기"입니다.

⚠️ 이 작업은 요약이 아닙니다.
⚠️ 문서를 줄이거나 압축하는 작업이 아닙니다.

[최우선 목표]
- 원문(raw_text)의 정보와 구조를 최대한 그대로 보존하여 JSON으로 구조화합니다.
- 원문에 없는 정보를 추가/추측/창작하지 않습니다.
- 원문에 있는 정보를 삭제하거나 생략하지 않습니다.
- 문서를 간략화하거나 압축하지 않습니다.
- 가능한 한 원문 정보량을 유지합니다.
- RAG 검색에 잘 걸리도록 구조만 정리합니다.

즉:
요약 ❌
재작성 ❌
구조화된 추출 ✅

[입력]
raw_text: {raw_text}

[출력 형식]
- RFC8259 유효 JSON만 출력
- 코드블록, 설명, 주석, 마크다운 금지
- 오직 JSON 텍스트만
- 모든 문자열은 큰따옴표 사용
- trailing comma 금지
- 모든 필드는 반드시 포함 (없으면 "" 또는 [])

{{
  "doc_title": "",
  "doc_type": "POLICY|PROCESS|FAQ|NOTICE|MIXED",
  "summary": "",
  "sections": [
    {{
      "section_id": "",
      "name": "",
      "tags": [],
      "content": [
        {{
          "rule_id": "",
          "title": "",
          "bullets": [],
          "structured": {{
            "target": "",
            "condition": "",
            "procedure": [],
            "exceptions": [],
            "owner": "",
            "channel": ""
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
  "change_summary": ""
}}

[🚨 절대 규칙 (위반 금지)]

1) 정보 삭제 금지
- 원문에 있는 항목/리스트/문장을 임의로 제거하지 마세요.

2) 정보 압축 금지
- 여러 항목을 하나로 합쳐 줄이지 마세요.

3) 요약 금지
- 의미만 남기고 축약하지 마세요.

4) 일반화 금지
- "등", "기타", "포함"으로 뭉개지 마세요.

5) 재구성 최소화
- 구조 정리 외 의미 변경 금지.

[섹션 규칙 - 최우선]

- sections는 원문 헤딩 구조를 그대로 따릅니다.
- 섹션 수를 임의로 늘리거나 줄이지 마세요.
- 병합/분리 금지.
- 원문 헤딩이 없으면 sections는 1개("general").

section_id:
- 섹션명 slug 사용
- 없으면 "general"

[rule 생성 규칙]

- 원문 불릿 리스트는 개수와 항목을 그대로 보존.
- 한 불릿 = 한 정보 단위.
- 삭제/병합 금지.
- "현재 버전", "웹사이트" 같은 정보도 각각 별도 rule 가능.

rule_id:
S1-R1, S1-R2… 순서 고정

[bullets 작성 규칙]

- 원문 문장을 최대한 유지
- 과도한 재작성 금지
- 정보 추가 금지

[source_quotes]

- 각 rule마다 0~2개
- 원문에서 그대로 복사
- 한국어 20~70자
- 없으면 []

[structured]

- 원문에 명시된 것만 채움
- 없으면 전부 빈값

[issues]

- 문제 없으면 []

[PII]

있으면:
- 마스킹
- issues에 PII_RISK
- pii_found=true

없으면:
- pii_found=false
- 나머지 []

[doc_type 판단]

POLICY: 규정/기준
PROCESS: 절차
FAQ: Q/A
NOTICE: 안내/소개
MIXED: 혼합

[마지막 체크 (스스로 검증)]

JSON 출력 전 반드시 확인:
- 원문 정보가 빠지지 않았는가?
- 불릿 개수가 줄지 않았는가?
- 요약하지 않았는가?

하나라도 위반이면 다시 생성하세요.

[금지]
- 추측
- 일반 상식 보완
- 새 정보 추가
- JSON 외 출력"""


@router.post("/doc/{doc_id}/manualize")
def manualize(doc_id: str, force: bool = False):
    """Convert raw text to structured manual sections.
    Default: single LLM call with full raw_text + MANUALIZE_PROMPT.
    Exception: if len(raw_text) > HARD_LIMIT, uses window-based fallback internally."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
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
        cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
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

    # Quick connectivity check before starting long operation
    llm_ok, llm_detail = ping_llm(timeout=5.0)
    if not llm_ok:
        conn.close()
        raise HTTPException(status_code=503, detail=llm_detail)

    raw_text_chars = len(raw_text)
    mode = get_llm_mode()
    window_count = len(_split_by_headings(raw_text))

    # 진행률 초기화 (sections: 완료된 섹션을 즉시 프론트에 스트리밍)
    _manualize_progress[doc_id] = {"done": 0, "total": window_count, "sections": {}}

    try:
        if mode == "remote":
            # Remote: 항상 섹션 분할 → 순차 처리 (내용 보존)
            print(f"[MANUALIZE] Remote sequential: {raw_text_chars} chars, {window_count} sections for {doc_id}")
            sections_map = _manualize_with_window(raw_text, doc_id)
        elif raw_text_chars > MANUALIZE_HARD_LIMIT:
            # Local + 대형 문서: 윈도우 병렬 처리
            print(f"[MANUALIZE] Local parallel: {raw_text_chars} chars > {MANUALIZE_HARD_LIMIT}")
            sections_map = _manualize_with_window(raw_text, doc_id)
        else:
            # Local + 소형 문서: 단일 호출
            print(f"[MANUALIZE] Local single: {raw_text_chars} chars for {doc_id}")
            sections_map = _manualize_single(raw_text, doc_id)
            _manualize_progress[doc_id] = {"done": 1, "total": 1, "sections": dict(sections_map)}

    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=502, detail=f"Manualize 실패: {str(e)}")

    # Check if cancelled
    if _manualize_progress.get(doc_id, {}).get("cancelled"):
        _manualize_progress.pop(doc_id, None)
        conn.close()
        raise HTTPException(status_code=499, detail="사용자가 Manualize를 취소했습니다.")

    # Save sections (gate_status stays NULL — user triggers Gate manually)
    cursor.execute("DELETE FROM manual_sections WHERE doc_id = %s", (doc_id,))
    cursor.execute("UPDATE documents SET completed_phases = NULL WHERE doc_id = %s", (doc_id,))
    sec_counter = 0
    for section_name, section_text in sections_map.items():
        sec_counter += 1
        if not section_name or not section_name.strip():
            section_name = f"## 기타 ({sec_counter})"
        section_id = f"sec_{uuid.uuid4().hex[:8]}"
        text_val = section_text if section_text else "정보 없음"
        cursor.execute(
            """INSERT INTO manual_sections
               (section_id, doc_id, section_name, section_text, ai_text,
                gate_status, gate_score, gate_reasons_json, gate_stale)
               VALUES (%s, %s, %s, %s, %s, NULL, NULL, NULL, 0)""",
            (section_id, doc_id, section_name, text_val, text_val)
        )

    cursor.execute("UPDATE documents SET updated_at = %s WHERE doc_id = %s",
                   (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "doc_id": doc_id,
        "sections": list(sections_map.keys()),
        "section_details": sections_map,
        "todo_questions": [],
        "change_summary": "",
        "pii_handling": {},
        "gate_results": {},
        "llm_used": True,
        "raw_text_chars": raw_text_chars,
        "window_count": window_count,
    }


def _manualize_single(raw_text: str, doc_id: str) -> dict:
    """Single LLM call with full raw_text. Returns {section_name: section_text}."""
    content = call_llm(MANUALIZE_PROMPT.format(raw_text=raw_text), temperature=0.3)
    if not content:
        raise Exception("LLM 응답이 비어있습니다.")

    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        raise Exception("LLM 응답에서 JSON을 찾을 수 없습니다.")

    parsed = _clean_llm_json(json_match.group())
    return _flatten_manualize_json(parsed)


def _split_by_headings(raw_text: str) -> list:
    """Split raw_text into sections by heading patterns.
    Returns list of (start_pos, section_text) tuples."""
    # Match common heading patterns: numbered (1. 2.1 제3조 etc.), markdown (#), or ALL-CAPS lines
    heading_pattern = re.compile(
        r'^(?:'
        r'#{1,4}\s+'                          # Markdown headings
        r'|제?\s*\d+[조항장절편]\s*'           # 법률/규정 스타일 (제1조, 제2장 등)
        r'|\d+(?:\.\d+)*[\.\)]\s+'            # Numbered (1. 2.1. 3.2.1)
        r'|[가-힣]{1,2}[\.\)]\s+'             # Korean bullets (가. 나.)
        r'|[IVX]+[\.\)]\s+'                   # Roman numerals
        r'|[A-Z][A-Z\s]{4,}$'                # ALL-CAPS lines (5+ chars)
        r')',
        re.MULTILINE
    )

    # Find all heading positions
    positions = [m.start() for m in heading_pattern.finditer(raw_text)]

    if not positions:
        # No headings found → return whole text as one section
        return [(0, raw_text)]

    # Ensure we start from 0
    if positions[0] != 0:
        positions.insert(0, 0)

    # Build sections
    sections = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(raw_text)
        sections.append((pos, raw_text[pos:end]))

    return sections


def _group_sections_into_windows(sections: list, window_size: int) -> list:
    """Group heading-based sections into windows that fit within window_size.
    Each window is a string. If a single section exceeds window_size, it is sent alone."""
    windows = []
    current_window = ""

    for _pos, section_text in sections:
        # If adding this section would exceed limit
        if current_window and len(current_window) + len(section_text) > window_size:
            windows.append(current_window)
            current_window = section_text
        else:
            current_window += section_text

    if current_window:
        windows.append(current_window)

    return windows


def _process_one_window(window_text: str, idx: int, total: int) -> dict:
    """Process a single window. Returns {section_name: section_text} or empty dict."""
    print(f"[MANUALIZE] Processing window {idx + 1}/{total} ({len(window_text)} chars)...")
    try:
        content = call_llm(MANUALIZE_PROMPT.format(raw_text=window_text), temperature=0.3)
        if not content:
            return {}
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return {}
        parsed = _clean_llm_json(json_match.group())
        return _flatten_manualize_json(parsed)
    except Exception as e:
        print(f"[MANUALIZE] Window {idx + 1} failed: {e}")
        return {}


def _merge_sections(all_sections: dict, new_sections: dict, idx: int) -> None:
    """Merge new_sections into all_sections. Dedup by exact match, index on conflict."""
    for name, text in new_sections.items():
        if name in all_sections:
            if all_sections[name].strip() == text.strip():
                continue
            name = f"{name} ({idx + 1})"
        all_sections[name] = text


def _manualize_with_window(raw_text: str, doc_id: str) -> dict:
    """Large document processing. Mode-aware:
    - Local: group into windows → parallel (concurrent threads)
    - Remote: heading sections → sequential (응답 받고 다음 전송)"""
    sections = _split_by_headings(raw_text)
    print(f"[MANUALIZE] Found {len(sections)} heading sections for {doc_id}")

    mode = get_llm_mode()

    if mode == "remote":
        # Remote: 섹션별 순차 처리 (작은 입력 → 내용 보존 우수)
        total = len(sections)
        print(f"[MANUALIZE] Remote sequential: {total} sections for {doc_id}")
        _manualize_progress[doc_id] = {"done": 0, "total": total, "sections": {}}
        all_sections = {}
        for i, (_pos, section_text) in enumerate(sections):
            # Check cancellation before each section
            if _manualize_progress.get(doc_id, {}).get("cancelled"):
                print(f"[MANUALIZE] Cancelled at section {i}/{total} for {doc_id}")
                break
            if not section_text.strip():
                _manualize_progress[doc_id]["done"] = i + 1
                continue
            result = _process_one_window(section_text, i, total)
            _merge_sections(all_sections, result, i)
            _manualize_progress[doc_id]["done"] = i + 1
            _manualize_progress[doc_id]["sections"] = dict(all_sections)
    else:
        # Local: 윈도우 묶어서 병렬 처리 (진행률은 섹션 수 기준으로 표시)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        windows = _group_sections_into_windows(sections, MANUALIZE_WINDOW_SIZE)
        total_windows = len(windows)
        total_sections = len(sections)
        print(f"[MANUALIZE] Local parallel: {total_windows} windows ({total_sections} sections) for {doc_id}")
        _manualize_progress[doc_id] = {"done": 0, "total": total_sections, "sections": {}}

        # Compute how many sections each window covers (mirrors _group_sections_into_windows logic)
        _sections_per_window = []
        _cur_len = 0
        _cur_cnt = 0
        for _pos, _st in sections:
            if _cur_len > 0 and _cur_len + len(_st) > MANUALIZE_WINDOW_SIZE:
                _sections_per_window.append(_cur_cnt)
                _cur_cnt = 1
                _cur_len = len(_st)
            else:
                _cur_cnt += 1
                _cur_len += len(_st)
        if _cur_cnt > 0:
            _sections_per_window.append(_cur_cnt)

        all_sections = {}
        if total_windows == 1:
            result = _process_one_window(windows[0], 0, 1)
            _merge_sections(all_sections, result, 0)
            _manualize_progress[doc_id]["done"] = total_sections
            _manualize_progress[doc_id]["sections"] = dict(all_sections)
        else:
            with ThreadPoolExecutor(max_workers=total_windows) as executor:
                futures = {
                    executor.submit(_process_one_window, w, i, total_windows): i
                    for i, w in enumerate(windows)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    _merge_sections(all_sections, result, i)
                    done_inc = _sections_per_window[i] if i < len(_sections_per_window) else 1
                    _manualize_progress[doc_id]["done"] = _manualize_progress[doc_id]["done"] + done_inc
                    _manualize_progress[doc_id]["sections"] = dict(all_sections)

    # 완료 후 정리
    _manualize_progress.pop(doc_id, None)

    if not all_sections:
        raise Exception("모든 처리가 실패했습니다.")

    return all_sections


# ============ STEP 2a: SPLIT (progressive) ============

@router.post("/doc/{doc_id}/split")
def split_document(doc_id: str, force: bool = False):
    """Split document into chunks (rule-based). Returns chunk list for progressive manualize.
    If not force and chunks already exist, returns cached chunks."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = doc["raw_text"]
    if not raw_text or raw_text.strip() == "" or raw_text.startswith("["):
        conn.close()
        raise HTTPException(status_code=400, detail="추출된 텍스트가 없습니다. 먼저 Extract를 수행해 주세요.")

    # Return cached chunks if not force
    if not force:
        cursor.execute("SELECT chunk_id, chunk_index, split_basis, notes, raw_chunk FROM doc_chunks WHERE doc_id = %s ORDER BY chunk_index", (doc_id,))
        existing = cursor.fetchall()
        if existing:
            chunks = []
            chunk_dicts = []
            for row in existing:
                preview = (row["raw_chunk"] or "")[:200]
                heading = row["notes"] or ""
                if not heading and row["raw_chunk"]:
                    first_line = row["raw_chunk"].split("\n", 1)[0].strip().lstrip("#").strip()
                    heading = first_line[:60] if first_line else f"Chunk {row['chunk_index'] + 1}"
                chunks.append({
                    "chunk_id": row["chunk_id"],
                    "chunk_index": row["chunk_index"],
                    "heading": heading,
                    "preview": preview,
                    "char_count": len(row["raw_chunk"] or ""),
                })
                chunk_dicts.append({
                    "chunk_id": row["chunk_id"],
                    "chunk_index": row["chunk_index"],
                    "notes": row["notes"] or "",
                    "raw_chunk": row["raw_chunk"] or "",
                })
            batches = _build_batches(chunk_dicts, doc_id)
            batch_info = [{"batch_id": b["batch_id"], "chunk_ids": b["chunk_ids"],
                           "oversize": [o["item_id"] for o in b.get("oversize_sections", [])]}
                          for b in batches]
            conn.close()
            return {"success": True, "doc_id": doc_id, "chunks": chunks, "batches": batch_info, "cached": True}

    # Split
    doc_chunks = _split_document(raw_text, doc_id, cursor)
    conn.commit()

    chunks = []
    for chunk in doc_chunks:
        raw_chunk = chunk.get("raw_chunk", "")
        heading = chunk.get("notes", "") or ""
        if not heading and raw_chunk:
            first_line = raw_chunk.split("\n", 1)[0].strip().lstrip("#").strip()
            heading = first_line[:60] if first_line else f"Chunk {chunk.get('chunk_index', 0) + 1}"
        chunks.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk.get("chunk_index", 0),
            "heading": heading,
            "preview": raw_chunk[:200],
            "char_count": len(raw_chunk),
        })

    batches = _build_batches(doc_chunks, doc_id)
    batch_info = [{"batch_id": b["batch_id"], "chunk_ids": b["chunk_ids"],
                   "oversize": [o["item_id"] for o in b.get("oversize_sections", [])]}
                  for b in batches]

    conn.close()
    return {"success": True, "doc_id": doc_id, "chunks": chunks, "batches": batch_info, "cached": False}


# ============ STEP 2b: MANUALIZE-CHUNK (progressive) ============

@router.post("/doc/{doc_id}/manualize-chunk/{chunk_id}")
def manualize_single_chunk(doc_id: str, chunk_id: str):
    """Manualize a single chunk. Returns section data immediately.
    Saves to manual_sections (appends, does not delete existing)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT chunk_id, chunk_index, raw_chunk, notes FROM doc_chunks WHERE chunk_id = %s AND doc_id = %s", (chunk_id, doc_id))
    chunk_row = cursor.fetchone()
    if not chunk_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Chunk not found")

    if not is_llm_available():
        conn.close()
        raise HTTPException(status_code=503, detail="LLM 비활성")

    chunk = {
        "chunk_id": chunk_row["chunk_id"],
        "chunk_index": chunk_row["chunk_index"],
        "raw_chunk": chunk_row["raw_chunk"] or "",
        "notes": chunk_row["notes"] or "",
    }

    result = _manualize_chunk(chunk, doc_id)

    section_name = result["section_name"]
    if not section_name or not section_name.strip():
        section_name = f"## 기타 (chunk-{chunk_id[:8]})"
    section_text = result["section_text"] or "정보 없음"
    evidence = result.get("evidence_spans", [])

    # Save section (check if already exists for this chunk)
    cursor.execute("SELECT section_id FROM manual_sections WHERE doc_id = %s AND source_chunk_id = %s", (doc_id, chunk_id))
    existing = cursor.fetchone()
    if existing:
        cursor.execute(
            """UPDATE manual_sections SET section_name = %s, section_text = %s,
               evidence_json = %s, updated_at = %s WHERE section_id = %s""",
            (section_name, section_text,
             json.dumps(evidence, ensure_ascii=False),
             datetime.now().isoformat(), existing["section_id"])
        )
        section_id = existing["section_id"]
    else:
        section_id = f"sec_{uuid.uuid4().hex[:8]}"
        cursor.execute(
            """INSERT INTO manual_sections
               (section_id, doc_id, section_name, section_text,
                source_chunk_id, evidence_json, merge_status)
               VALUES (%s, %s, %s, %s, %s, %s, 'BODY')""",
            (section_id, doc_id, section_name, section_text,
             chunk_id, json.dumps(evidence, ensure_ascii=False))
        )

    cursor.execute("UPDATE documents SET updated_at = %s WHERE doc_id = %s",
                   (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()

    evidence_matched = sum(1 for s in evidence if s.get("char_start", -1) >= 0)

    return {
        "success": result.get("success", True),
        "chunk_id": chunk_id,
        "section_id": section_id,
        "section_name": section_name,
        "section_text": section_text,
        "evidence_spans": evidence,
        "evidence_matched": evidence_matched,
        "evidence_total": len(evidence),
    }


# ============ STEP 2c: MANUALIZE-BATCH (progressive) ============

@router.post("/doc/{doc_id}/manualize-batch/{batch_id}")
def manualize_batch_endpoint(doc_id: str, batch_id: str):
    """Manualize a batch of chunks in a single LLM call.
    Returns section data for all items in the batch."""
    conn = get_connection()
    cursor = conn.cursor()

    # Load chunks for this batch
    cursor.execute("SELECT chunk_id, chunk_index, notes, raw_chunk FROM doc_chunks WHERE doc_id = %s ORDER BY chunk_index", (doc_id,))
    all_chunks = [dict(row) for row in cursor.fetchall()]
    if not all_chunks:
        conn.close()
        raise HTTPException(status_code=404, detail="No chunks found")

    # Rebuild batches to find the requested one
    batches = _build_batches(all_chunks, doc_id)
    target_batch = None
    for b in batches:
        if b["batch_id"] == batch_id:
            target_batch = b
            break
    if not target_batch:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    if not is_llm_available():
        conn.close()
        raise HTTPException(status_code=503, detail="LLM 비활성")

    # Call LLM with batch prompt
    results = _manualize_batch(target_batch, doc_id)

    # Build chunk_id → raw_chunk map for evidence indexing
    chunk_map = {c["chunk_id"]: c["raw_chunk"] for c in all_chunks}

    # Save each result to manual_sections
    saved = []
    for item in target_batch["items"]:
        chunk_id = item["item_id"]
        # Find matching result
        result = None
        for r in results:
            if r.get("item_id") == chunk_id:
                result = r
                break
        if not result:
            # Fallback: use first unmatched result or raw text
            if results:
                result = results.pop(0)
            else:
                result = {
                    "section_name": item["section_title"] or f"Chunk {chunk_id}",
                    "section_text": item["section_text"][:3000],
                    "evidence_spans": [],
                }

        section_name = result.get("section_name", item["section_title"] or f"Chunk {chunk_id}")
        if not section_name or not section_name.strip():
            section_name = f"## 기타 (chunk-{chunk_id[:8]})"
        section_text = result.get("section_text", "") or "정보 없음"
        evidence_spans = result.get("evidence_spans", [])

        # Index evidence spans
        raw_chunk = chunk_map.get(chunk_id, "")
        indexed_spans = _index_evidence_spans(evidence_spans, raw_chunk)

        # Save/update
        cursor.execute("SELECT section_id FROM manual_sections WHERE doc_id = %s AND source_chunk_id = %s", (doc_id, chunk_id))
        existing = cursor.fetchone()
        if existing:
            cursor.execute(
                """UPDATE manual_sections SET section_name = %s, section_text = %s,
                   evidence_json = %s, updated_at = %s WHERE section_id = %s""",
                (section_name, section_text,
                 json.dumps(indexed_spans, ensure_ascii=False),
                 datetime.now().isoformat(), existing["section_id"]))
            section_id = existing["section_id"]
        else:
            section_id = f"sec_{uuid.uuid4().hex[:8]}"
            cursor.execute(
                """INSERT INTO manual_sections
                   (section_id, doc_id, section_name, section_text,
                    source_chunk_id, evidence_json, merge_status)
                   VALUES (%s, %s, %s, %s, %s, %s, 'BODY')""",
                (section_id, doc_id, section_name, section_text,
                 chunk_id, json.dumps(indexed_spans, ensure_ascii=False)))

        evidence_matched = sum(1 for s in indexed_spans if s.get("char_start", -1) >= 0)
        saved.append({
            "chunk_id": chunk_id,
            "section_id": section_id,
            "section_name": section_name,
            "section_text": section_text,
            "evidence_spans": indexed_spans,
            "evidence_matched": evidence_matched,
            "evidence_total": len(indexed_spans),
        })

    # Handle oversize sections
    oversize_errors = []
    for o in target_batch.get("oversize_sections", []):
        oversize_errors.append({
            "chunk_id": o["item_id"],
            "code": "OVERSIZE_SECTION",
            "message": f"섹션이 {o['char_len']}자로 배치 한도({MAX_BATCH_CHARS}자) 초과",
        })

    cursor.execute("UPDATE documents SET updated_at = %s WHERE doc_id = %s",
                   (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "batch_id": batch_id,
        "sections": saved,
        "errors": oversize_errors,
    }


# ============ STEP 2: SECTION GATE (per-section AI check) ============

GATE_CHECK_PROMPT = """당신은 RAG 반영 전, 매뉴얼 섹션 텍스트(section_text)의 품질/리스크를 판정하는 QA 게이트입니다.
중요: 내용을 새로 작성하거나 고치지 말고, 오직 '검증 결과'만 출력하세요.

[입력]
section_text: {section_text}
raw_text: {raw_text}

────────────────
[판정 상태]
────────────────
PASS: 바로 RAG 반영 가능
NEED_FIX: 사람이 수정/보강 후 반영 권장
BLOCK: RAG 반영 금지(보안/심각 충돌/대량 유실)

────────────────
[🚨 최우선 원칙]
────────────────
- 추측 금지.
- 모든 지적은 section_text 또는 raw_text의 **증거 문자열 기반**이어야 함.
- location_hint에 증거를 제시할 수 없으면 reasons에 넣지 마세요.
- "가능성", "추정", "아마" 같은 표현 기반 지적 금지.
- 같은 내용을 반복/과잉 지적하지 마세요(핵심만).

────────────────
[운영 안전핀]
────────────────
- HIGH severity는 "명백하고 직접적인 증거"가 있을 때만 부여하세요.
  애매하거나 해석 여지가 있으면 HIGH로 올리지 말고 MEDIUM으로 두세요(과도한 BLOCK 방지).
- HIGH는 일반 해석이 아니라, 사람이 보아도 명백한 위반/충돌로 인식되는 경우에만 부여하세요.
- score는 정교한 평가가 아니라 status 구간(PASS/NEED_FIX/BLOCK)을 반영하는 대략적 값입니다.
  운영 판단은 score가 아니라 status를 우선하세요.

────────────────
[검증 항목]
────────────────

1) PII_RISK (BLOCK 우선 — 실제 값일 때만)
다음 **실제 값**이 마스킹 없이 있을 때만 문제:
- 주민번호 패턴
- 계좌/ID성 긴 숫자열(10~16자리)

PII_RISK가 아닌 것 (예외):
- 고객센터/대표 전화번호 (02-xxxx-xxxx, 1588-xxxx 등): 공개 연락처이므로 PII 아님
- 서비스 안내용 이메일/URL: 공개 정보이므로 PII 아님
- 개인 휴대폰(010-xxxx-xxxx)도 매뉴얼에 의도적으로 기재된 것이면 PII 아님
- "전화번호/이메일/주소" 같은 필드명은 PII 아님

────────────────

2) CONFLICT
동일 조건/대상/상황에서 **직접 충돌**하는 규칙만 해당.
(가능 vs 불가, 허용 vs 금지 등)
location_hint: 충돌하는 문장 중 하나를 section_text에서 **원문 그대로** 복사 (10~60자)

────────────────

3) MISSING / AMBIGUOUS
실행 규칙인데 필수값(기한/금액/조건/채널/담당) 없으면 MISSING.
모호 표현 반복 시 AMBIGUOUS:
(적당히/가능하면/상황에 따라/협의 후/필요시)
소개/개요 섹션에는 과도 적용 금지.

────────────────

4) HALLUCINATION_RISK
section_text에 수치/기간/금액/의무 규칙이 있는데,
raw_text에서 **동일 주제 관련 표현 전반**을 찾기 어려울 때만 해당.
- 단일 키워드 부재로 판단 금지.
- section_text에서 핵심 키워드 1~3개를 먼저 뽑아 비교.

예외 (HALLUCINATION_RISK로 판정하지 말 것):
- 연락처(전화번호), 이메일, URL, 웹사이트 주소: 사용자가 의도적으로 추가했을 수 있음
- "지원", "문의", "안내" 등 서비스 안내 섹션의 연락 정보는 신뢰
- [사용자 직접 수정 구간]으로 표시된 내용

location_hint:
- section_text에서 문제 되는 문장/불릿을 **원문 그대로** 복사 (10~60자)

severity:
- 기본 MEDIUM
- HIGH는 서로 다른 구체 수치/기간/금액/의무 규정이 2개 이상 추가된 경우만
  (그리고 그 추가가 명백히 확인될 때만)

────────────────

5) FORMAT
FORMAT은 아래일 때만:
- section_text가 "##"로 시작하지 않음
- ### 항목이 0개 AND '- ' bullet도 3개 미만

예외:
### 0개라도 '- ' bullet ≥ 3이면 FORMAT 아님.

FORMAT 단독 BLOCK 금지.
(형식 문제는 최대 NEED_FIX까지만)

────────────────

6) OMISSION (정보 유실) — 방향: raw_text → section_text만 해당
raw_text의 **실행 정보**가 section_text에 없을 때만 OMISSION.
적용 대상:
- 절차 / 조건 / 예외 / 금액 / 기한 / 담당 / 채널
적용 제외:
- 배경 설명 / 홍보 문구 / 예시/부연

⚠️ 방향 주의:
- OMISSION = raw_text에 있는데 section_text에 없는 것
- section_text에 있는데 raw_text에 없는 것은 OMISSION이 아님 (→ HALLUCINATION_RISK 검토 대상)
- section_text에만 존재하는 연락처/URL/이메일 등을 OMISSION으로 분류하지 마세요.
- [사용자가 직접 삭제한 내용]으로 표시된 내용은 OMISSION으로 판정하지 마세요.

판정(증거 필수):
- raw_text에서 누락 예시 2개(짧은 문장/불릿) 제시
- section_text에서 해당 내용이 없음을 명시

location_hint: 누락된 내용이 추가되어야 할 section_text의 **인접 줄**을 원문 그대로 복사 (10~60자).
(예: raw_text에 '락커 종류: 소형 40x40cm'가 누락 → section_text의 '락커(사물함)설정' 줄을 복사)

대량 누락(핵심 실행정보 다수 누락) 시만 BLOCK.
- "대량 누락"은 실행 정보 유형(절차/조건/예외/금액/기한/담당/채널) 중 **2가지 이상 유형에 대해**
  raw_text에 명시된 실행 정보가 section_text에서 **반복적으로 누락**되는 경우를 의미합니다.
  (단, 단일 항목 수준의 경미 누락은 HIGH로 올리지 말고 MEDIUM으로 두세요.)

────────────────
[검증 불충분]
────────────────
PASS 금지 → NEED_FIX (INSUFFICIENT_EVIDENCE)

해당 조건(반드시 근거로 설명):
- raw_text < 800자 AND (OMISSION/CONFLICT/HALLUCINATION_RISK) 대조에 필요한 근거가 부족함
- section_text 핵심 키워드(1~3개)를 raw_text에서 찾기 어려워 근거 대조가 사실상 불가능함
- 근거 부족으로 판단 불가

location_hint: section_text에서 근거 부족한 문장을 **원문 그대로** 복사 (10~60자)

────────────────
[출력 형식]
────────────────
RFC8259 JSON만 출력.

제약:
- reasons는 최대 6개까지만 출력
- 같은 type은 최대 2개까지만 출력

{{
  "status": "PASS|NEED_FIX|BLOCK",
  "score": 0,
  "reasons": [
    {{
      "type": "PII_RISK|CONFLICT|MISSING|AMBIGUOUS|HALLUCINATION_RISK|FORMAT|OMISSION|INSUFFICIENT_EVIDENCE",
      "severity": "LOW|MEDIUM|HIGH",
      "message": "문제 설명",
      "location_hint": "section_text에서 문제가 되는 부분을 **그대로 복사**하세요 (10~60자). 원문과 한 글자라도 다르면 하이라이트가 안 됩니다. 설명/사유는 여기에 쓰지 마세요.",
      "fix_suggestion": "수정 방법"
    }}
  ],
  "required_actions": []
}}

────────────────
[Score 기준]
────────────────
PASS: 85~100
NEED_FIX: 50~84
BLOCK: 0~49

감점 기준(100에서 감점):
HIGH = -40
MEDIUM = -20
LOW = -10

- score는 0~100 정수로 클램프(범위를 벗어나면 0 또는 100으로 보정)

────────────────
[Status 결정 규칙]
────────────────
아래 조건이 reasons에 존재하면 status는 반드시 그렇게 결정해야 합니다(강제).

- PII_RISK(HIGH) → BLOCK (무조건)
- CONFLICT(HIGH) → BLOCK
- OMISSION(HIGH) → BLOCK
- HALLUCINATION_RISK(HIGH) → BLOCK

추가 규칙:
- HIGH 1개(위 4개 유형 외 포함) → 최소 NEED_FIX
- MEDIUM 2개 이상 → NEED_FIX
- LOW만 존재 → PASS 가능
- FORMAT은 BLOCK 금지 유지

────────────────
[PASS 최종 조건]
────────────────
reasons가 비어 있어도 아래 모두 만족:
- PII 없음
- FORMAT 통과
- 검증 충분(INSUFFICIENT_EVIDENCE 아님)
- 명백한 누락/환각 없음
"""


def _gate_section_internal(doc_id: str, section_name: str, cursor, raw_text_fallback: str = None):
    """Core gate logic reusable by gate_section() and gate-all.
    cursor must be from an open connection. Does NOT commit/close.
    raw_text_fallback: optional pre-fetched raw_text to avoid redundant query."""
    import time as _time
    t_start = _time.time()

    cursor.execute("SELECT section_id, section_text, source_chunk_id, ai_text, gate_reasons_json FROM manual_sections WHERE doc_id = %s AND section_name = %s",
                   (doc_id, section_name))
    sec = cursor.fetchone()
    if not sec:
        return {"success": False, "section_name": section_name, "error": "Section not found"}

    # Load previously dismissed reasons
    prev_dismissed = []
    try:
        prev_reasons = json.loads(sec["gate_reasons_json"] or "[]")
        prev_dismissed = [r for r in prev_reasons if r.get("dismissed")]
    except Exception:
        pass

    # Determine raw reference: prefer raw_chunk from doc_chunks if available
    raw_reference = ""
    if sec["source_chunk_id"]:
        cursor.execute("SELECT raw_chunk FROM doc_chunks WHERE chunk_id = %s", (sec["source_chunk_id"],))
        chunk_row = cursor.fetchone()
        if chunk_row and chunk_row["raw_chunk"]:
            raw_reference = chunk_row["raw_chunk"][:4000]

    if not raw_reference:
        if raw_text_fallback is not None:
            raw_reference = raw_text_fallback[:4000]
        else:
            cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
            doc = cursor.fetchone()
            raw_reference = (doc["raw_text"] or "")[:4000] if doc else ""

    # Detect user edits: diff section_text vs ai_text
    user_edits_note = ""
    ai_text = sec["ai_text"] or ""
    section_text = sec["section_text"] or ""
    if ai_text and section_text != ai_text:
        ai_lines = set(ai_text.strip().splitlines())
        sec_lines = set(section_text.strip().splitlines())
        sec_lines_list = section_text.strip().splitlines()
        # 사용자가 추가한 라인 (ai_text에 없고 section_text에 있음)
        added = [l for l in sec_lines_list if l.strip() and l not in ai_lines]
        # 사용자가 삭제한 라인 (ai_text에 있고 section_text에 없음)
        removed = [l for l in ai_text.strip().splitlines() if l.strip() and l not in sec_lines]
        parts = []
        if added:
            added_preview = "\n".join(added[:10])
            parts.append(f"[사용자가 직접 추가한 내용 — HALLUCINATION_RISK 제외 대상]\n{added_preview}\n(위 내용은 사용자가 의도적으로 추가한 것이므로 raw_text에 없더라도 HALLUCINATION_RISK로 판정하지 마세요.)")
        if removed:
            removed_preview = "\n".join(removed[:10])
            parts.append(f"[사용자가 직접 삭제한 내용 — OMISSION 제외 대상]\n{removed_preview}\n(위 내용은 사용자가 의도적으로 삭제한 것이므로 section_text에 없더라도 OMISSION으로 판정하지 마세요.)")
        if parts:
            user_edits_note = "\n\n" + "\n\n".join(parts)

    # Build dismissed note for LLM prompt
    dismissed_note = ""
    if prev_dismissed:
        dismissed_items = []
        for d in prev_dismissed:
            dismissed_items.append(f"- [{d.get('type','?')}] {d.get('description','')[:100]}")
        dismissed_note = "\n\n[이전 검증에서 사용자가 무시(dismiss)한 이슈 - 동일 이슈를 다시 보고하지 마세요]\n" + "\n".join(dismissed_items)

    gate_result = {"status": "PASS", "score": 100, "reasons": [], "required_actions": []}

    if is_llm_available():
        try:
            prompt_text = GATE_CHECK_PROMPT.format(
                section_text=section_text[:3000],
                raw_text=raw_reference
            ) + user_edits_note + dismissed_note
            print(f"[GATE_SECTION] calling LLM for '{section_name}' (user_edits_note: {len(user_edits_note)} chars, dismissed: {len(prev_dismissed)})")
            content = call_llm(prompt_text, temperature=0.3)
            print(f"[GATE_SECTION] LLM response for '{section_name}': {content[:300] if content else 'EMPTY'}")
            if content:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    gate_result = _clean_llm_json(json_match.group())
                    print(f"[GATE_SECTION] result for '{section_name}': status={gate_result.get('status')}, reasons={len(gate_result.get('reasons', []))}")
        except Exception as e:
            print(f"[GATE_SECTION] LLM error: {e}")
    else:
        print(f"[GATE_SECTION] LLM not available, returning default PASS")

    # Post-process: restore dismissed flags for matching reasons
    new_reasons = gate_result.get("reasons", [])
    if prev_dismissed:
        dismissed_keys = {(d.get("type", ""), d.get("description", "")[:80]) for d in prev_dismissed}
        for r in new_reasons:
            key = (r.get("type", ""), r.get("description", "")[:80])
            if key in dismissed_keys:
                r["dismissed"] = True

    # Append previously dismissed reasons that LLM didn't re-report (keep them visible as dismissed)
    existing_keys = {(r.get("type", ""), r.get("description", "")[:80]) for r in new_reasons}
    for d in prev_dismissed:
        key = (d.get("type", ""), d.get("description", "")[:80])
        if key not in existing_keys:
            d["dismissed"] = True
            new_reasons.append(d)

    gate_result["reasons"] = new_reasons

    # Recalculate status from active (non-dismissed) reasons only
    active = [r for r in new_reasons if not r.get("dismissed")]
    if not active:
        gate_result["status"] = "PASS"
        gate_result["score"] = max(gate_result.get("score", 100), 85)
    else:
        has_high = any(r.get("severity") == "HIGH" for r in active)
        high_types = {r.get("type") for r in active if r.get("severity") == "HIGH"}
        block_types = {"PII_RISK", "CONFLICT", "OMISSION", "HALLUCINATION_RISK"}
        if has_high and high_types & block_types:
            gate_result["status"] = "BLOCK"
        elif has_high or sum(1 for r in active if r.get("severity") == "MEDIUM") >= 2:
            gate_result["status"] = "NEED_FIX"
        else:
            gate_result["status"] = "PASS"

    # Save gate result to manual_sections
    cursor.execute(
        "UPDATE manual_sections SET gate_status = %s, gate_score = %s, gate_reasons_json = %s, gate_stale = 0, updated_at = %s WHERE section_id = %s",
        (gate_result["status"], gate_result.get("score", 100),
         json.dumps(gate_result["reasons"], ensure_ascii=False),
         datetime.now().isoformat(), sec["section_id"])
    )

    gate_result["time_s"] = round(_time.time() - t_start, 2)
    return {"success": True, "section_name": section_name, **gate_result}


@router.post("/doc/{doc_id}/section/{section_name}/gate")
def gate_section(doc_id: str, section_name: str):
    """Run AI gate check on a single section.
    If source_chunk_id exists, use raw_chunk instead of full raw_text (chunk-based)."""
    conn = get_connection()
    cursor = conn.cursor()
    result = _gate_section_internal(doc_id, section_name, cursor)
    if not result.get("success"):
        conn.close()
        raise HTTPException(status_code=404, detail=result.get("error", "Section not found"))
    conn.commit()
    conn.close()
    return result


@router.post("/doc/{doc_id}/gate-all")
def gate_all(doc_id: str):
    """Run AI gate check on ALL sections of a document."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT section_name FROM manual_sections WHERE doc_id = %s", (doc_id,))
    sections = cursor.fetchall()
    if not sections:
        conn.close()
        raise HTTPException(status_code=400, detail="No sections found")

    # Pre-fetch raw_text once for fallback
    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    doc = cursor.fetchone()
    raw_text_fallback = (doc["raw_text"] or "") if doc else ""

    results = []
    pass_count = need_fix_count = block_count = 0
    for sec in sections:
        r = _gate_section_internal(doc_id, sec["section_name"], cursor, raw_text_fallback=raw_text_fallback)
        results.append(r)
        status = r.get("status", "PASS")
        if status == "PASS":
            pass_count += 1
        elif status == "NEED_FIX":
            need_fix_count += 1
        elif status == "BLOCK":
            block_count += 1

    conn.commit()
    conn.close()
    return {
        "success": True, "pass_count": pass_count,
        "need_fix_count": need_fix_count, "block_count": block_count,
        "results": results
    }


@router.post("/doc/{doc_id}/fill-all")
def fill_all(doc_id: str):
    """맥락 보강(Fill)만 전 섹션에 실행."""
    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM 사용 불가")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    raw_text = doc["raw_text"] or ""
    raw_text_safe = raw_text[:4000] if raw_text else "(원본 문서를 찾을 수 없습니다. 기존 텍스트만 참고하세요.)"

    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
    all_sections = cursor.fetchall()
    if not all_sections:
        conn.close()
        raise HTTPException(status_code=400, detail="No sections found")

    total = len(all_sections)
    done_count = 0
    failed_sections = []

    for sec in all_sections:
        section_name = sec["section_name"]
        try:
            qa_policy_text = "Q&A는 새로 추가하지 마세요. 기존 Q&A만 유지/정리하세요."
            prompt = FILL_SECTION_TEXT_PROMPT_V3.format(
                section_text=sec["section_text"],
                raw_text=raw_text_safe,
                qa_policy_text=qa_policy_text
            )
            result = call_llm(prompt, temperature=0.3)
            if result and result.strip():
                cursor.execute(
                    "UPDATE manual_sections SET section_text = %s, updated_at = %s WHERE doc_id = %s AND section_name = %s",
                    (result.strip(), datetime.now().isoformat(), doc_id, section_name)
                )
                done_count += 1
        except Exception as e:
            print(f"[FILL_ALL] error for {section_name}: {e}")
            failed_sections.append({"section_name": section_name, "reason": str(e)})

    conn.commit()
    conn.close()
    return {
        "success": True, "total": total,
        "done_count": done_count, "failed_count": len(failed_sections),
        "failed_sections": failed_sections
    }


@router.post("/doc/{doc_id}/refine-all")
def refine_all(doc_id: str):
    """RAG 최적화만 전 섹션에 실행."""
    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM 사용 불가")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    raw_text = doc["raw_text"] or ""
    raw_text_safe = raw_text[:4000] if raw_text else "(원본 없음)"

    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
    all_sections = cursor.fetchall()
    if not all_sections:
        conn.close()
        raise HTTPException(status_code=400, detail="No sections found")

    total = len(all_sections)
    done_count = 0
    failed_sections = []

    for sec in all_sections:
        section_name = sec["section_name"]
        try:
            prompt = FINALIZE_SECTION_TEXT_PROMPT_V1.format(
                section_text=sec["section_text"],
                raw_text=raw_text_safe
            )
            result = call_llm(prompt, temperature=0.3)
            if result and result.strip():
                cursor.execute(
                    "UPDATE manual_sections SET section_text = %s, updated_at = %s WHERE doc_id = %s AND section_name = %s",
                    (result.strip(), datetime.now().isoformat(), doc_id, section_name)
                )
                done_count += 1
        except Exception as e:
            print(f"[REFINE_ALL] error for {section_name}: {e}")
            failed_sections.append({"section_name": section_name, "reason": str(e)})

    conn.commit()
    conn.close()
    return {
        "success": True, "total": total,
        "done_count": done_count, "failed_count": len(failed_sections),
        "failed_sections": failed_sections
    }


class PhaseUpdate(BaseModel):
    phase: str  # "fill", "refine", "gate"


@router.post("/doc/{doc_id}/completed-phase")
def save_completed_phase(doc_id: str, req: PhaseUpdate):
    """Add a completed phase to the document's completed_phases field."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT completed_phases FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    existing = row["completed_phases"] or ""
    phases = [p for p in existing.split(",") if p]
    if req.phase not in phases:
        phases.append(req.phase)
    cursor.execute("UPDATE documents SET completed_phases = %s WHERE doc_id = %s",
                   (",".join(phases), doc_id))
    conn.commit()
    conn.close()
    return {"success": True, "completed_phases": ",".join(phases)}


@router.post("/doc/{doc_id}/snapshot-ai-text")
def snapshot_ai_text(doc_id: str):
    """Copy current section_text to ai_text for all sections (after AI operation)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE manual_sections SET ai_text = section_text WHERE doc_id = %s",
        (doc_id,)
    )
    updated = cursor.rowcount
    conn.commit()
    conn.close()
    return {"success": True, "updated": updated}


class SnapshotSectionAiText(BaseModel):
    section_name: str


@router.post("/doc/{doc_id}/snapshot-ai-text-section")
def snapshot_ai_text_section(doc_id: str, req: SnapshotSectionAiText):
    """Copy current section_text to ai_text for a single section."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE manual_sections SET ai_text = section_text WHERE doc_id = %s AND section_name = %s",
        (doc_id, req.section_name)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@router.post("/doc/{doc_id}/section/{section_name}/gate-dismiss")
def gate_dismiss(doc_id: str, section_name: str, body: dict = Body(...)):
    """Toggle dismissed flag on a gate reason. Recalculate status from active reasons."""
    reason_index = body.get("reason_index")
    if reason_index is None:
        raise HTTPException(status_code=400, detail="reason_index required")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT gate_reasons_json, gate_status FROM manual_sections WHERE doc_id = %s AND section_name = %s",
        (doc_id, section_name)
    )
    sec = cursor.fetchone()
    if not sec:
        conn.close()
        raise HTTPException(status_code=404, detail="Section not found")
    reasons = []
    try:
        reasons = json.loads(sec["gate_reasons_json"] or "[]")
    except Exception:
        pass
    if not (0 <= reason_index < len(reasons)):
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid reason_index")
    # Toggle dismissed flag
    reasons[reason_index]["dismissed"] = not reasons[reason_index].get("dismissed", False)
    # Recalculate status from active (non-dismissed) reasons only
    active = [r for r in reasons if not r.get("dismissed")]
    if not active:
        new_status = "PASS"
    else:
        has_high = any(r.get("severity") == "HIGH" for r in active)
        high_types = {r.get("type") for r in active if r.get("severity") == "HIGH"}
        block_types = {"PII_RISK", "CONFLICT", "OMISSION", "HALLUCINATION_RISK"}
        if has_high and high_types & block_types:
            new_status = "BLOCK"
        else:
            new_status = "NEED_FIX"
    cursor.execute(
        "UPDATE manual_sections SET gate_reasons_json = %s, gate_status = %s, gate_stale = 0, updated_at = %s WHERE doc_id = %s AND section_name = %s",
        (json.dumps(reasons, ensure_ascii=False), new_status, datetime.now().isoformat(), doc_id, section_name)
    )
    conn.commit()
    conn.close()
    return {"success": True, "status": new_status, "reasons": reasons}


@router.post("/doc/{doc_id}/section/{section_name}/gate-stale")
def set_gate_stale(doc_id: str, section_name: str):
    """Mark section as gate_stale (saved without gate re-check)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE manual_sections SET gate_stale = 1, updated_at = %s WHERE doc_id = %s AND section_name = %s",
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
    
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
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
                    llm_issues = _clean_llm_json(json_match.group())
                    issues.extend(llm_issues[:5])  # Limit LLM issues
            else:
                llm_error_msg = "LLM 응답이 비어있습니다. (API 할당량 초과 가능성)"
        except Exception as e:
            llm_error_msg = f"LLM 호출 실패: {str(e)}"
            print(f"[QUALITY_GATE] LLM error: {e}")
    
    # Clear old issues and save new
    cursor.execute("DELETE FROM qa_issues WHERE doc_id = %s", (doc_id,))
    for issue in issues:
        issue_id = f"issue_{uuid.uuid4().hex[:8]}"
        cursor.execute(
            "INSERT INTO qa_issues (issue_id, doc_id, severity, issue_type, message, suggestion, status) VALUES (%s, %s, %s, %s, %s, %s, 'OPEN')",
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
            "UPDATE manual_sections SET section_text = %s WHERE doc_id = %s AND section_name = %s",
            (text, doc_id, name)
        )
    
    cursor.execute("UPDATE documents SET updated_at = %s WHERE doc_id = %s", (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()
    
    return {"success": True, "doc_id": doc_id}


# ============ STEP 2: AI HELPER (Fill/Refine) ============

class RefineRequest(BaseModel):
    text: str
    task: str  # "refine", "fill", "recommend"
    context: Optional[str] = None  # Section name or issue message
    allow_qa: Optional[str] = None  # "true" or "false" (for fill task)


def _clean_llm_json(raw: str):
    """Clean and parse JSON from LLM response. Handles common issues.
    Returns dict or list depending on input."""
    raw = re.sub(r',\s*([}\]])', r'\1', raw)       # trailing comma
    raw = raw.replace('\n', ' ').replace('\r', '')   # newlines
    raw = re.sub(r'[\x00-\x1f]', ' ', raw)          # control chars
    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Attempt 2: single → double quotes
    try:
        return json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        pass
    # Attempt 3: fix unescaped quotes inside string values
    # e.g. "message": "문제는 "이것" 입니다" → "message": "문제는 \"이것\" 입니다"
    try:
        fixed = re.sub(
            r'(?<=: )"((?:[^"\\]|\\.)*?)"(?=\s*[,}\]])',
            lambda m: '"' + m.group(1) + '"',
            raw
        )
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Attempt 4: try closing at each } or ] from the end (handles trailing garbage)
    close_positions = [i for i, c in enumerate(raw) if c in ('}', ']')]
    for pos in reversed(close_positions):
        try:
            return json.loads(raw[:pos + 1])
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("_clean_llm_json: all attempts failed", raw, 0)


def _flatten_manualize_json(parsed: dict) -> dict:
    """Convert MANUALIZE_PROMPT JSON output to flat {section_name: section_text} dict.
    Formats each section as structured markdown (## / ### / - bullets)."""
    sections_map = {}
    for sec in parsed.get("sections", []):
        name = sec.get("name", sec.get("section_id", "general"))
        lines = [f"## {name}"]
        for rule in sec.get("content", []):
            title = rule.get("title", "")
            if title:
                lines.append(f"### {title}")
            for bullet in rule.get("bullets", []):
                lines.append(f"- {bullet}")
            # structured fields
            st = rule.get("structured", {})
            for key in ("target", "condition", "owner", "channel"):
                val = st.get(key, "")
                if val:
                    lines.append(f"- {key}: {val}")
            for proc_item in st.get("procedure", []):
                lines.append(f"- {proc_item}")
            for exc_item in st.get("exceptions", []):
                lines.append(f"- 예외: {exc_item}")
        section_text = "\n".join(lines)
        sections_map[name] = section_text
    # Fallback: if no sections parsed, try summary
    if not sections_map:
        summary = parsed.get("summary", "")
        if summary:
            sections_map["general"] = f"## 요약\n- {summary}"
        else:
            sections_map["general"] = "정보 없음"
    return sections_map


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


# ============ CHUNK-BASED PIPELINE: PROMPTS ============

SPLIT_PROMPT = """당신은 긴 문서를 RAG 처리에 적합한 chunk(청크)로 분할하는 전문가입니다.

[입력]
raw_text: {raw_text}

[규칙]
1) 헤딩(#, ##, 숫자. 등) 경계를 우선으로 분할합니다.
2) 각 chunk는 3,000~8,000자 목표, 최대 12,000자를 넘지 않습니다.
3) 3,000자 미만의 짧은 섹션은 인접 섹션과 합칩니다.
4) start_anchor, end_anchor는 원문에서 그대로 복사한 문자열(20~60자)입니다.
   - start_anchor: 해당 chunk가 시작하는 원문 문장/헤딩의 첫 부분
   - end_anchor: 해당 chunk가 끝나는 원문 문장의 마지막 부분
5) split_basis: 분할 근거 (예: "heading", "length", "topic_shift")
6) chunk_id는 "C1", "C2"... 순서입니다.

[출력 형식]
RFC8259 JSON 배열만 출력. 코드블록/설명/주석 금지.

[
  {{
    "chunk_id": "C1",
    "start_anchor": "원문 시작 부분 텍스트",
    "end_anchor": "원문 끝 부분 텍스트",
    "split_basis": "heading",
    "notes": ""
  }}
]

[제약]
- anchor는 반드시 원문(raw_text)에 존재하는 문자열이어야 합니다.
- chunk 간 겹침 없이 전체 문서를 빠짐없이 커버해야 합니다.
- JSON 배열 외 어떤 출력도 금지합니다."""

MANUALIZE_CHUNK_PROMPT = """당신은 영업/운영 문서의 일부(chunk)를 RAG에 넣기 위한
"구조화된 정보 추출기"입니다.

⚠️ 이 작업은 요약이 아닙니다. 문서를 줄이거나 압축하는 작업이 아닙니다.

[최우선 목표]
- 입력된 chunk(raw_chunk)의 정보와 구조를 최대한 그대로 보존하여 구조화합니다.
- 원문에 없는 정보를 추가/추측/창작하지 않습니다.
- 원문에 있는 정보를 삭제하거나 생략하지 않습니다.
- 개인정보(전화번호, 이메일, 계좌번호 등)는 ***로 마스킹합니다.

[입력]
raw_chunk: {raw_chunk}

[출력 형식]
RFC8259 유효 JSON만 출력. 코드블록, 설명, 주석, 마크다운 금지.

{{
  "section_name": "이 chunk의 주제를 대표하는 섹션명",
  "section_text": "구조화된 텍스트 (## 헤딩, ### 항목, - 불릿 형식)",
  "evidence_spans": [
    {{
      "span_text": "raw_chunk에서 복사한 근거 문장 (20~80자)",
      "maps_to": "section_text 내 대응 위치 설명 (항목명 또는 불릿 요약)",
      "is_pii": false
    }}
  ]
}}

[evidence_spans 규칙]
- 최소 2개 이상 작성 (가능하면 핵심 정보마다 1개)
- span_text는 raw_chunk 원문에서 그대로 복사 (verbatim)
- is_pii=true인 span은 마스킹된 형태로 작성
- maps_to는 section_text의 어느 부분에 대응하는지 간단히 표시

[section_text 작성 규칙]
- "## 섹션명"으로 시작
- "### 항목제목"으로 하위 구분
- "- " 불릿으로 본문
- 원문 정보 삭제/요약/압축 금지
- 원문 불릿 개수와 항목 그대로 보존

[금지]
- 추측, 일반 상식 보완, 새 정보 추가
- JSON 외 출력"""


# ============ CHUNK-BASED PIPELINE: HELPERS ============

def _resolve_anchors(raw_text: str, chunks_raw: list) -> list:
    """Resolve start_anchor/end_anchor to char_start/char_end positions in raw_text."""
    resolved = []
    search_from = 0
    for i, chunk in enumerate(chunks_raw):
        start_anchor = chunk.get("start_anchor", "")
        end_anchor = chunk.get("end_anchor", "")

        # Find start position
        char_start = raw_text.find(start_anchor, search_from) if start_anchor else search_from
        if char_start < 0:
            # Fallback: try from beginning
            char_start = raw_text.find(start_anchor) if start_anchor else search_from
        if char_start < 0:
            char_start = search_from

        # Find end position
        if end_anchor:
            end_search = max(char_start, search_from)
            end_pos = raw_text.find(end_anchor, end_search)
            if end_pos >= 0:
                char_end = end_pos + len(end_anchor)
            else:
                # Fallback: next chunk's start or end of text
                char_end = len(raw_text) if i == len(chunks_raw) - 1 else None
        else:
            char_end = len(raw_text) if i == len(chunks_raw) - 1 else None

        # Deferred end for non-last chunks
        if char_end is None:
            char_end = len(raw_text)  # will be adjusted by next chunk

        resolved.append({
            **chunk,
            "char_start": char_start,
            "char_end": char_end,
        })
        search_from = char_start + 1

    # Adjust: ensure no gaps/overlaps between consecutive chunks
    for i in range(len(resolved) - 1):
        if resolved[i + 1]["char_start"] < resolved[i]["char_end"]:
            resolved[i]["char_end"] = resolved[i + 1]["char_start"]

    # Extract raw_chunk text
    for r in resolved:
        r["raw_chunk"] = raw_text[r["char_start"]:r["char_end"]]

    return resolved


def _save_doc_chunks(doc_id: str, chunks: list, cursor) -> list:
    """Save resolved chunks to doc_chunks table. Returns list with chunk_ids."""
    cursor.execute("DELETE FROM doc_chunks WHERE doc_id = %s", (doc_id,))
    saved = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"dchunk_{uuid.uuid4().hex[:8]}"
        cursor.execute("""
            INSERT INTO doc_chunks (chunk_id, doc_id, chunk_index, start_anchor, end_anchor,
                                    char_start, char_end, split_basis, notes, raw_chunk)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (chunk_id, doc_id, i,
              chunk.get("start_anchor", ""), chunk.get("end_anchor", ""),
              chunk.get("char_start", 0), chunk.get("char_end", 0),
              chunk.get("split_basis", ""), chunk.get("notes", ""),
              chunk.get("raw_chunk", "")))
        saved.append({**chunk, "chunk_id": chunk_id, "chunk_index": i})
    return saved


def _index_evidence_spans(spans: list, raw_chunk: str) -> list:
    """Index span_text positions within raw_chunk. Sets char_start/char_end or -1 if not found."""
    indexed = []
    for span in spans:
        span_text = span.get("span_text", "")
        is_pii = span.get("is_pii", False)

        if is_pii or not span_text:
            indexed.append({**span, "char_start": -1, "char_end": -1})
            continue

        pos = raw_chunk.find(span_text)
        if pos < 0:
            # Fallback: case-insensitive
            pos = raw_chunk.lower().find(span_text.lower())
        if pos >= 0:
            indexed.append({**span, "char_start": pos, "char_end": pos + len(span_text)})
        else:
            indexed.append({**span, "char_start": -1, "char_end": -1})
    return indexed


def _conservative_dedup(sections: list) -> list:
    """Remove exact-duplicate sections by section_name. Keep first occurrence."""
    seen = set()
    result = []
    for sec in sections:
        name = sec.get("section_name", "")
        if name not in seen:
            seen.add(name)
            result.append(sec)
    return result


# ============ BATCH MANUALIZE ============

MAX_BATCH_CHARS = 12000

MANUALIZE_BATCH_PROMPT = """당신은 'Manualize 배치 실행기'입니다.
중요: 아래 [MANUALIZE_RULES]의 규칙을 **그대로 준수**하여 작업하세요.

────────────────
[MANUALIZE_RULES - SOURCE OF TRUTH]
────────────────
- 원문(section_text)의 정보와 구조를 최대한 그대로 보존하여 구조화합니다.
- 원문에 없는 정보를 추가/추측/창작하지 않습니다.
- 원문에 있는 정보를 삭제하거나 생략하지 않습니다.
- 요약/압축/일반화 금지.
- 개인정보(전화번호, 이메일, 계좌번호 등)는 ***로 마스킹합니다.
- section_text 내부의 헤딩/번호/구분 구조를 그대로 매핑하세요.
- 구분이 없으면 sections=1개(general)로 처리하세요.
- 출력 형식: "## 섹션명", "### 항목제목", "- " 불릿.
- evidence_spans: 최소 2개, span_text는 원문에서 그대로 복사(verbatim).

────────────────
[입력]
────────────────
{batches_json}

────────────────
[작업 규칙 - 배치 전용]
────────────────
1) item(=섹션) 간 정보 섞기 금지. 각 섹션은 독립적으로 Manualize.
2) 섹션 구조 보존. 섹션 수를 임의 조정 금지.
3) oversize_sections에 있는 항목은 Manualize 하지 말고 errors에 기록만.

────────────────
[출력 형식]
────────────────
RFC8259 유효 JSON만 출력. 코드블록/설명/주석 금지.

{{
  "batch_id": "<입력 batch_id>",
  "results": [
    {{
      "item_id": "S001",
      "section_name": "섹션 주제를 대표하는 이름",
      "section_text": "구조화된 텍스트 (## / ### / - 형식)",
      "evidence_spans": [
        {{
          "span_text": "원문에서 복사한 근거 문장 (20~80자)",
          "maps_to": "section_text 내 대응 위치",
          "is_pii": false
        }}
      ]
    }}
  ],
  "errors": []
}}"""


def _build_batches(doc_chunks: list, doc_id: str, max_chars: int = MAX_BATCH_CHARS) -> list:
    """Group doc_chunks into batches. Section boundary = batch boundary.
    A section never splits across batches. If adding a section exceeds max_chars,
    it goes to the next batch."""
    batches = []
    current_items = []
    current_chars = 0
    oversize = []
    batch_index = 0

    for chunk in doc_chunks:
        raw_chunk = chunk.get("raw_chunk", "")
        char_len = len(raw_chunk)
        heading = chunk.get("notes", "") or ""
        if not heading and raw_chunk:
            first_line = raw_chunk.split("\n", 1)[0].strip().lstrip("#").strip()
            heading = first_line[:60] if first_line else f"Chunk {chunk.get('chunk_index', 0) + 1}"

        item = {
            "item_id": chunk["chunk_id"],
            "section_title": heading,
            "section_text": raw_chunk,
            "char_len": char_len,
        }

        # Oversize: single section exceeds max_chars
        if char_len > max_chars:
            oversize.append({
                "item_id": chunk["chunk_id"],
                "section_title": heading,
                "char_len": char_len,
                "reason": "section_text exceeds max_batch_chars"
            })
            continue

        # Would exceed limit → close current batch, start new one
        if current_items and current_chars + char_len > max_chars:
            batch_index += 1
            batches.append({
                "doc_id": doc_id,
                "batch_id": f"B{batch_index:03d}",
                "batch_index": batch_index - 1,
                "max_batch_chars": max_chars,
                "items": current_items,
                "oversize_sections": [],
                "chunk_ids": [it["item_id"] for it in current_items],
            })
            current_items = []
            current_chars = 0

        current_items.append(item)
        current_chars += char_len

    # Last batch
    if current_items:
        batch_index += 1
        batches.append({
            "doc_id": doc_id,
            "batch_id": f"B{batch_index:03d}",
            "batch_index": batch_index - 1,
            "max_batch_chars": max_chars,
            "items": current_items,
            "oversize_sections": [],
            "chunk_ids": [it["item_id"] for it in current_items],
        })

    # Attach oversize to first batch (or create one)
    if oversize:
        if batches:
            batches[0]["oversize_sections"] = oversize
        else:
            batches.append({
                "doc_id": doc_id,
                "batch_id": "B001",
                "batch_index": 0,
                "max_batch_chars": max_chars,
                "items": [],
                "oversize_sections": oversize,
                "chunk_ids": [],
            })

    return batches


def _manualize_batch(batch: dict, doc_id: str) -> list:
    """Process a single batch via LLM. Returns list of per-item results."""
    items = batch.get("items", [])
    if not items:
        return []

    # Build batch JSON for prompt
    batch_input = {
        "doc_id": doc_id,
        "batch_id": batch["batch_id"],
        "max_batch_chars": batch["max_batch_chars"],
        "items": [
            {"item_id": it["item_id"], "section_title": it["section_title"],
             "section_text": it["section_text"], "char_len": it["char_len"]}
            for it in items
        ],
        "oversize_sections": batch.get("oversize_sections", []),
    }

    for attempt in range(2):
        try:
            content = call_llm(
                MANUALIZE_BATCH_PROMPT.format(
                    batches_json=json.dumps(batch_input, ensure_ascii=False)
                ),
                temperature=0.3
            )
            if content:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = _clean_llm_json(json_match.group())
                    results = parsed.get("results", [])
                    if results:
                        return results
        except Exception as e:
            print(f"[MANUALIZE_BATCH] attempt {attempt + 1} failed for {batch['batch_id']}: {e}")

    # Fallback: return raw text as-is for each item
    fallback = []
    for it in items:
        fallback.append({
            "item_id": it["item_id"],
            "section_name": it["section_title"] or f"[미처리] {it['item_id']}",
            "section_text": it["section_text"][:3000],
            "evidence_spans": [],
        })
    return fallback


# ============ CHUNK-BASED PIPELINE: CORE FUNCTIONS ============

def _split_document(raw_text: str, doc_id: str, cursor):
    """Split document into chunks using rule-based splitting (no LLM).
    Returns list of resolved chunks with raw_chunk text."""
    chunks_raw = []

    # Rule-based: heading split
    heading_chunks = _split_raw_by_headings(raw_text)
    if heading_chunks:
        for i, hc in enumerate(heading_chunks):
            start_anchor = hc["body"][:50].strip() if hc["body"] else ""
            end_anchor = hc["body"][-50:].strip() if hc["body"] else ""
            chunks_raw.append({
                "chunk_id": f"C{i + 1}",
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "split_basis": "heading",
                "notes": hc["heading"]
            })

    # Fallback: single chunk (entire document)
    if not chunks_raw:
        print(f"[SPLIT] No headings found, single chunk for {doc_id}")
        chunks_raw = [{
            "chunk_id": "C1",
            "start_anchor": raw_text[:50].strip(),
            "end_anchor": raw_text[-50:].strip(),
            "split_basis": "single_document",
            "notes": "No split possible"
        }]

    # Resolve anchors to char positions and extract raw_chunk
    resolved = _resolve_anchors(raw_text, chunks_raw)

    # Save to DB
    saved = _save_doc_chunks(doc_id, resolved, cursor)
    print(f"[SPLIT] {doc_id}: {len(saved)} chunks created")
    return saved


def _manualize_chunk(chunk: dict, doc_id: str) -> dict:
    """Manualize a single chunk. Returns dict with section_name, section_text, evidence_spans, chunk_id.
    Retry once on failure."""
    raw_chunk = chunk.get("raw_chunk", "")
    chunk_id = chunk.get("chunk_id", "")

    for attempt in range(2):
        try:
            content = call_llm(
                MANUALIZE_CHUNK_PROMPT.format(raw_chunk=raw_chunk[:10000]),
                temperature=0.3
            )
            if content:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result = _clean_llm_json(json_match.group())
                    section_name = result.get("section_name", f"Chunk-{chunk.get('chunk_index', 0)}")
                    section_text = result.get("section_text", "")
                    evidence_spans = result.get("evidence_spans", [])

                    # Index evidence spans
                    indexed_spans = _index_evidence_spans(evidence_spans, raw_chunk)

                    return {
                        "section_name": section_name,
                        "section_text": section_text,
                        "evidence_spans": indexed_spans,
                        "chunk_id": chunk_id,
                        "success": True
                    }
        except Exception as e:
            print(f"[MANUALIZE_CHUNK] attempt {attempt + 1} failed for {chunk_id}: {e}")

    # Both attempts failed — return raw chunk as-is
    return {
        "section_name": f"[미처리] Chunk-{chunk.get('chunk_index', 0)}",
        "section_text": raw_chunk[:3000],
        "evidence_spans": [],
        "chunk_id": chunk_id,
        "success": False
    }


def _gate_chunk(section_text: str, raw_chunk: str) -> dict:
    """Run gate check on a chunk using only the raw_chunk (not full raw_text)."""
    gate_result = {"status": "PASS", "score": 100, "reasons": [], "required_actions": []}

    if not is_llm_available():
        return gate_result

    try:
        content = call_llm(GATE_CHECK_PROMPT.format(
            section_text=section_text[:3000],
            raw_text=raw_chunk[:4000]
        ), temperature=0.3)
        if content:
            gm = re.search(r'\{[\s\S]*\}', content)
            if gm:
                gate_result = _clean_llm_json(gm.group())
    except Exception as e:
        print(f"[GATE_CHUNK] error: {e}")

    return gate_result


def _merge_chunks(chunk_results: list, gate_results: list) -> dict:
    """Merge chunk manualize results. PASS → BODY, NEED_FIX → APPENDIX.
    Returns {sections_map, gate_map, evidence_map, merge_info}."""
    sections_map = {}
    gate_map = {}
    evidence_map = {}
    merge_info = {}

    for i, result in enumerate(chunk_results):
        gate = gate_results[i] if i < len(gate_results) else {"status": "PASS", "score": 100, "reasons": []}
        name = result["section_name"]
        gate_status = gate.get("status", "PASS")

        # Determine merge_status
        if gate_status in ("NEED_FIX", "BLOCK"):
            merge_status = "APPENDIX"
            display_name = f"[부록] {name}"
        else:
            merge_status = "BODY"
            display_name = name

        # Conservative dedup: if same name exists, append index
        if display_name in sections_map:
            display_name = f"{display_name} ({i + 1})"

        sections_map[display_name] = result["section_text"]
        gate_map[display_name] = gate
        evidence_map[display_name] = result.get("evidence_spans", [])
        merge_info[display_name] = {
            "merge_status": merge_status,
            "chunk_id": result.get("chunk_id", ""),
            "original_name": name,
        }

    return {
        "sections_map": sections_map,
        "gate_map": gate_map,
        "evidence_map": evidence_map,
        "merge_info": merge_info,
    }


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

⚠️ 당신이 다듬어야 하는 대상은 아래 [section_text]뿐입니다.
⚠️ [raw_text]는 원본 참고용입니다. raw_text의 다른 섹션 내용을 section_text에 추가하지 마세요.
⚠️ 입력된 section_text 범위 밖의 내용은 절대 포함하지 마세요.

[입력]
section_text (다듬을 대상 — 이 섹션만 출력): {section_text}

raw_text (참고용 원본 — 출력하지 마세요): {raw_text}

[절대 규칙]
1) 이 섹션(section_text)만 다듬어 출력. 다른 섹션 내용 포함 금지
2) 새 정보 추가 금지(수치/기간/금액/정책/예외/절차 창작 금지)
3) 원문/현 section_text와 다른 사실 생성 금지
4) 개인정보 마스킹 유지(***)
5) 근거가 불명확한 문장 추가 금지. 불명확하면 "[확인 필요: ...]"를 유지하거나 더 명확히 작성

[최적화 목표]
- 검색 키워드에 잘 걸리도록 항목 제목(###)을 '질문형 또는 키워드형'으로 선명하게
  예: "환불 규정" → "환불 규정/기한/위약금"
- bullets를 짧고 병렬 구조로 정리(중복 제거)
- 같은 의미의 표현을 통일(용어 표준화)
- Q&A가 이미 존재하면, 질문을 더 명확히 하되 답은 그대로(내용 추가 금지)
- 너무 긴 bullet은 2개로 분리하되 의미 유지

[출력]
- 오직 이 섹션의 최종 section_text만 plain text로 출력하세요.
- 다른 섹션 내용을 합쳐서 출력하지 마세요.
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
    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
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
    """Approve document if no RED issues open and no unresolved evidence failures."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT doc_id FROM documents WHERE doc_id = %s", (doc_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    cursor.execute("SELECT COUNT(*) as cnt FROM qa_issues WHERE doc_id = %s AND severity = 'RED' AND status = 'OPEN'", (doc_id,))
    red_count = cursor.fetchone()["cnt"]

    if red_count > 0:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Cannot approve: {red_count} RED issues open")

    # Check evidence span indexing failures (non-PII spans with char_start=-1)
    cursor.execute("SELECT section_name, evidence_json FROM manual_sections WHERE doc_id = %s AND evidence_json IS NOT NULL", (doc_id,))
    evidence_failures = []
    for row in cursor.fetchall():
        try:
            spans = json.loads(row["evidence_json"] or "[]")
            for span in spans:
                if span.get("char_start", 0) == -1 and not span.get("is_pii", False):
                    evidence_failures.append(row["section_name"])
                    break
        except (json.JSONDecodeError, TypeError):
            pass

    if evidence_failures:
        conn.close()
        sections_str = ", ".join(evidence_failures[:3])
        raise HTTPException(
            status_code=400,
            detail=f"Evidence span 매칭 실패 섹션 {len(evidence_failures)}개 ({sections_str}). Manualize를 다시 실행하세요."
        )

    cursor.execute("UPDATE documents SET status = 'APPROVED', updated_at = %s WHERE doc_id = %s",
                   (datetime.now().isoformat(), doc_id))
    conn.commit()
    conn.close()

    # Auto-reindex after approval
    reindex_result = reindex(doc_id)

    return {"success": True, "doc_id": doc_id, "status": "APPROVED", "reindex": reindex_result}


@router.post("/doc/{doc_id}/publish-all")
def publish_all(doc_id: str):
    """Index PASS sections only into RAG chunks. Update doc status to APPROVED.
    RAG optimize is already done in fill-all, so this just chunks and indexes."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT section_name, section_text, gate_status FROM manual_sections WHERE doc_id = %s", (doc_id,))
    all_sections = cursor.fetchall()
    if not all_sections:
        conn.close()
        raise HTTPException(status_code=400, detail="No sections found")

    pass_sections = [s for s in all_sections if s["gate_status"] == "PASS"]
    excluded_count = len(all_sections) - len(pass_sections)

    published_count = 0
    failed_count = 0
    failed_sections = []

    # Delete existing chunks for this doc (will re-insert PASS only)
    cursor.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    chunk_size = 500
    overlap = 100

    for sec in pass_sections:
        section_name = sec["section_name"]
        text = sec["section_text"]

        try:
            if not text or text == "정보 없음":
                continue

            start = 0
            chunk_index = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                if chunk_text.strip():
                    chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
                    # Generate embedding
                    embedding_blob = None
                    try:
                        emb = get_embedding(chunk_text)
                        embedding_blob = np.array(emb, dtype=np.float32).tobytes()
                    except Exception as emb_err:
                        print(f"[PUBLISH_ALL] Embedding failed for {chunk_id}: {emb_err}")
                    cursor.execute(
                        "INSERT INTO chunks (chunk_id, doc_id, section_name, chunk_index, chunk_text, embedding, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (chunk_id, doc_id, section_name, chunk_index, chunk_text, embedding_blob, datetime.now().isoformat())
                    )
                    chunk_index += 1

                start = end - overlap
                if start >= len(text):
                    break

            published_count += 1
        except Exception as e:
            print(f"[PUBLISH_ALL] Indexing failed for {section_name}: {e}")
            failed_count += 1
            failed_sections.append({"section_name": section_name, "reason": str(e)})

    # Update doc status to APPROVED
    cursor.execute("UPDATE documents SET status = 'APPROVED', updated_at = %s WHERE doc_id = %s",
                   (datetime.now().isoformat(), doc_id))

    conn.commit()
    conn.close()

    # Build FAISS index for this apt_id
    embed_stats = {}
    try:
        conn2 = get_connection()
        c2 = conn2.cursor()
        c2.execute("SELECT apt_id FROM documents WHERE doc_id = %s", (doc_id,))
        doc_row = c2.fetchone()
        conn2.close()
        if doc_row:
            embed_stats = rag_build_index(doc_row["apt_id"])
    except Exception as idx_err:
        print(f"[PUBLISH_ALL] FAISS index build failed: {idx_err}")
        embed_stats = {"error": str(idx_err)}

    published_names = [s["section_name"] for s in pass_sections if s["section_name"] not in [f["section_name"] for f in failed_sections]]
    excluded_names = [s["section_name"] for s in all_sections if s["gate_status"] != "PASS"]
    return {
        "success": True,
        "published_count": published_count,
        "failed_count": failed_count,
        "excluded_count": excluded_count,
        "failed_sections": failed_sections,
        "published_names": published_names,
        "excluded_names": excluded_names,
        "embed_stats": embed_stats
    }


# ============ STEP 2: REINDEX ============

@router.post("/doc/{doc_id}/reindex")
def reindex(doc_id: str):
    """Chunk manual sections for RAG retrieval."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
    sections = cursor.fetchall()
    if not sections:
        conn.close()
        raise HTTPException(status_code=400, detail="No sections to index")
    
    # Delete existing chunks
    cursor.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    
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
                # Generate embedding
                embedding_blob = None
                try:
                    emb = get_embedding(chunk_text)
                    embedding_blob = np.array(emb, dtype=np.float32).tobytes()
                except Exception as emb_err:
                    print(f"[REINDEX] Embedding failed for {chunk_id}: {emb_err}")
                cursor.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, section_name, chunk_index, chunk_text, embedding, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (chunk_id, doc_id, section_name, chunk_index, chunk_text, embedding_blob, datetime.now().isoformat())
                )
                chunk_count += 1
                chunk_index += 1

            start = end - overlap
            if start >= len(text):
                break

    conn.commit()
    conn.close()

    # Build FAISS index for this apt_id
    embed_stats = {}
    try:
        conn2 = get_connection()
        c2 = conn2.cursor()
        c2.execute("SELECT apt_id FROM documents WHERE doc_id = %s", (doc_id,))
        doc_row = c2.fetchone()
        conn2.close()
        if doc_row:
            embed_stats = rag_build_index(doc_row["apt_id"])
    except Exception as idx_err:
        print(f"[REINDEX] FAISS index build failed: {idx_err}")
        embed_stats = {"error": str(idx_err)}

    return {"success": True, "doc_id": doc_id, "chunk_count": chunk_count, "embed_stats": embed_stats}


# ============ STEP 2: GET SECTIONS ============

@router.get("/doc/{doc_id}/sections")
def get_sections(doc_id: str):
    """Get manual sections for a document (includes chunk-based fields)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""SELECT section_name, section_text, gate_status, gate_reasons_json, gate_stale,
                             evidence_json, source_chunk_id, merge_status
                      FROM manual_sections WHERE doc_id = %s""", (doc_id,))
    sections = [dict(row) for row in cursor.fetchall()]
    # Include document-level completed_phases
    cursor.execute("SELECT completed_phases FROM documents WHERE doc_id = %s", (doc_id,))
    doc_row = cursor.fetchone()
    completed_phases = (doc_row["completed_phases"] or "") if doc_row else ""
    conn.close()
    return {"sections": sections, "completed_phases": completed_phases}


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


def _body_fallback_match(section_text: str, raw_text: str, min_phrase_len: int = 6) -> str | None:
    """Fallback: find the best matching region in raw_text using keyword phrases from section_text.
    Returns the matched raw_text region or None."""
    if not section_text or not raw_text:
        return None

    # Extract meaningful phrases (lines stripped of markdown markers)
    sec_lines = [ln.lstrip("#>*- \t").strip() for ln in section_text.split("\n")]
    sec_lines = [ln for ln in sec_lines if len(ln) >= min_phrase_len]
    if not sec_lines:
        return None

    # Find matching lines in raw_text
    raw_lines = raw_text.split("\n")
    matched_raw_indices = set()
    for sec_ln in sec_lines:
        # Try exact substring match first
        sec_ln_clean = sec_ln.lstrip("#>*- \t").strip()
        if len(sec_ln_clean) < min_phrase_len:
            continue
        for ri, raw_ln in enumerate(raw_lines):
            raw_ln_clean = raw_ln.lstrip("#>*- \t").strip()
            # Check if significant portion overlaps
            if len(raw_ln_clean) < 4:
                continue
            # Extract keywords (3+ chars) from section line
            sec_words = set(re.findall(r'[\w가-힣]{3,}', sec_ln_clean.lower()))
            raw_words = set(re.findall(r'[\w가-힣]{3,}', raw_ln_clean.lower()))
            if not sec_words:
                continue
            overlap = len(sec_words & raw_words) / len(sec_words)
            if overlap >= 0.5:
                matched_raw_indices.add(ri)

    if not matched_raw_indices:
        return None

    # Build contiguous region: from first matched line to last matched line (with 1-line padding)
    first = max(0, min(matched_raw_indices) - 1)
    last = min(len(raw_lines) - 1, max(matched_raw_indices) + 1)
    region = "\n".join(raw_lines[first:last + 1]).strip()
    return region if region else None


@router.get("/doc/{doc_id}/source-map")
def get_source_map(doc_id: str):
    """Get per-section matched raw text.
    Chunk-based: uses doc_chunks JOIN for precise raw_chunk mapping.
    Legacy: falls back to heading matching."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT raw_text FROM documents WHERE doc_id = %s", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    cursor.execute("SELECT section_name, source_chunk_id, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
    sections = cursor.fetchall()
    conn.close()

    raw_text = doc["raw_text"] or ""

    # Check if any section has source_chunk_id (chunk-based document)
    has_chunks = any(row["source_chunk_id"] for row in sections)

    if has_chunks:
        # Chunk-based: JOIN doc_chunks for raw_chunk
        conn2 = get_connection()
        cursor2 = conn2.cursor()
        source_map = {}
        matched_count = 0
        for row in sections:
            name = row["section_name"]
            chunk_id = row["source_chunk_id"]
            if chunk_id:
                cursor2.execute("SELECT raw_chunk FROM doc_chunks WHERE chunk_id = %s", (chunk_id,))
                chunk_row = cursor2.fetchone()
                if chunk_row and chunk_row["raw_chunk"]:
                    source_map[name] = chunk_row["raw_chunk"]
                    matched_count += 1
                else:
                    source_map[name] = None
            else:
                source_map[name] = None
        conn2.close()
        # Body fallback for unmatched sections
        for row in sections:
            name = row["section_name"]
            if source_map.get(name) is None:
                fallback = _body_fallback_match(row["section_text"] or "", raw_text)
                if fallback:
                    source_map[name] = fallback
                    matched_count += 1
        return {"source_map": source_map, "raw_text": raw_text, "matched": matched_count}

    # Legacy: heading-based matching
    section_names = [row["section_name"] for row in sections]
    section_texts = {row["section_name"]: row["section_text"] or "" for row in sections}
    chunks = _split_raw_by_headings(raw_text)
    if not chunks:
        # No headings found — try body fallback for every section
        source_map = {}
        for name in section_names:
            source_map[name] = _body_fallback_match(section_texts[name], raw_text)
        return {"source_map": source_map, "raw_text": raw_text, "matched": sum(1 for v in source_map.values() if v is not None)}

    match_indices = []
    for sec_name in section_names:
        idx = _heading_match(sec_name, chunks)
        match_indices.append(idx)

    source_map = {}
    for i, sec_name in enumerate(section_names):
        idx = match_indices[i]
        if idx < 0:
            # Heading match failed — try body fallback
            source_map[sec_name] = _body_fallback_match(section_texts[sec_name], raw_text)
            continue
        next_chunk_idx = len(chunks)
        for j in range(i + 1, len(section_names)):
            if match_indices[j] > idx:
                next_chunk_idx = match_indices[j]
                break
        parts = [chunks[k]["body"] for k in range(idx, min(next_chunk_idx, len(chunks)))]
        source_map[sec_name] = "\n\n".join(parts)

    return {"source_map": source_map, "raw_text": raw_text, "matched": sum(1 for v in source_map.values() if v is not None)}


@router.get("/doc/{doc_id}/issues")
def get_issues(doc_id: str):
    """Get QA issues for a document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT issue_id, severity, issue_type, message, suggestion, status FROM qa_issues WHERE doc_id = %s", (doc_id,))
    issues = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return issues


# ============ STEP 3: CHAT WITH RAG ============

class ChatRequest(BaseModel):
    apt_id: str
    client_id: str = "default"
    conversation_id: Optional[str] = None
    message: str
    model: Optional[str] = None


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
    import time as _time
    t_start = _time.time()
    timing = {}

    conn = get_connection()
    cursor = conn.cursor()

    # Get or create conversation
    if req.conversation_id:
        conv_id = req.conversation_id
        cursor.execute("UPDATE conversations SET last_at = %s WHERE conversation_id = %s",
                       (datetime.now().isoformat(), conv_id))
    else:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        cursor.execute(
            "INSERT INTO conversations (conversation_id, apt_id, client_id, created_at, last_at) VALUES (%s, %s, %s, %s, %s)",
            (conv_id, req.apt_id, req.client_id, datetime.now().isoformat(), datetime.now().isoformat())
        )
    
    # Save user message
    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    cursor.execute(
        "INSERT INTO messages (msg_id, conversation_id, role, text, created_at) VALUES (%s, %s, 'user', %s, %s)",
        (user_msg_id, conv_id, req.message, datetime.now().isoformat())
    )
    
    # FAISS vector search with keyword fallback
    t_rag = _time.time()
    search_method = "faiss"
    top_chunks = []
    try:
        top_chunks = faiss_search(req.apt_id, req.message, top_k=5)
    except Exception as faiss_err:
        print(f"[CHAT] FAISS search failed, falling back to keyword: {faiss_err}")

    if not top_chunks:
        # Fallback: keyword search
        search_method = "keyword"
        cursor.execute("""
            SELECT c.chunk_id, c.doc_id, c.section_name, c.chunk_text, d.title as doc_title
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE d.apt_id = %s AND d.status = 'APPROVED'
        """, (req.apt_id,))
        all_chunks = [dict(row) for row in cursor.fetchall()]
        top_chunks = keyword_search(req.message, all_chunks, top_k=5)
    timing["rag_s"] = round(_time.time() - t_rag, 2)
    
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
                t_llm = _time.time()
                content = call_llm(CHAT_PROMPT.format(context=context, question=req.message), temperature=0.3, model=req.model)
                timing["llm_s"] = round(_time.time() - t_llm, 2)
                if content:
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        parsed = _clean_llm_json(json_match.group())
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
        "INSERT INTO messages (msg_id, conversation_id, role, text, meta_json, created_at) VALUES (%s, %s, 'assistant', %s, %s, %s)",
        (asst_msg_id, conv_id, response["reply_text"], meta_json, datetime.now().isoformat())
    )
    
    conn.commit()
    conn.close()

    timing["total_s"] = round(_time.time() - t_start, 2)
    response["timing"] = timing
    response["search_method"] = search_method

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
        WHERE c.apt_id = %s AND m.role = 'assistant'
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
        "SELECT doc_id FROM documents WHERE apt_id = %s AND status = 'APPROVED' ORDER BY version DESC LIMIT 1",
        (apt_id,)
    )
    doc_row = cursor.fetchone()
    target_doc_id = doc_row["doc_id"] if doc_row else None
    
    # Save suggestions
    for sug in suggestions:
        sug_id = f"sug_{uuid.uuid4().hex[:8]}"
        cursor.execute("""
            INSERT INTO improve_suggestions (sug_id, apt_id, title, reason, proposed_patch, target_doc_id, target_section_name, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'PENDING', %s, %s)
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
    cursor.execute("SELECT * FROM improve_suggestions WHERE sug_id = %s", (sug_id,))
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
        "SELECT section_id, section_text FROM manual_sections WHERE doc_id = %s AND section_name = %s",
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
        "UPDATE manual_sections SET section_text = %s WHERE section_id = %s",
        (new_text, section_id)
    )
    
    # Update suggestion status
    cursor.execute(
        "UPDATE improve_suggestions SET status = 'APPLIED', updated_at = %s WHERE sug_id = %s",
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
    cursor.execute("SELECT section_name, section_text FROM manual_sections WHERE doc_id = %s", (doc_id,))
    sections = cursor.fetchall()
    
    # Get API_NEEDED issues
    cursor.execute("SELECT message FROM qa_issues WHERE doc_id = %s AND issue_type = 'API_NEEDED'", (doc_id,))
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
                    specs = _clean_llm_json(json_match.group())
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
    cursor.execute("DELETE FROM api_specs WHERE doc_id = %s", (doc_id,))
    for spec in specs:
        spec_id = f"spec_{uuid.uuid4().hex[:8]}"
        cursor.execute("""
            INSERT INTO api_specs (spec_id, doc_id, intent, endpoint, method, req_fields_json, resp_fields_json, auth, errors_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        FROM api_specs WHERE doc_id = %s
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
            WHERE s.apt_id = %s
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
        cursor.execute("SELECT * FROM conversations WHERE apt_id = %s ORDER BY last_at DESC", (apt_id,))
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
    cursor.execute("SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at", (conversation_id,))
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
            REPLACE INTO branch_class_cache
            (apt_id, branch_id, class_id, name, start, end, capacity, reserved, asof, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            WHERE branch_id = %s AND apt_id = %s AND start LIKE %s
            ORDER BY start
        """, (branch_id, apt_id, f"{date}%"))
    else:
        cursor.execute("""
            SELECT * FROM branch_class_cache
            WHERE branch_id = %s AND apt_id = %s
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
        cursor.execute("SELECT * FROM branch_class_cache WHERE apt_id = %s ORDER BY start", (apt_id,))
    else:
        cursor.execute("SELECT * FROM branch_class_cache ORDER BY start")
    
    classes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"classes": classes, "count": len(classes)}
