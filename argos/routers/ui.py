"""UI routes for web pages."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from database import get_connection

router = APIRouter(tags=["ui"])

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=RedirectResponse)
def home():
    """Redirect to upload page."""
    return RedirectResponse(url="/ui/upload", status_code=302)


@router.get("/ui/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    """Screen A: Upload & Manualize."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get apartments
    cursor.execute("SELECT apt_id, name FROM apartments ORDER BY name")
    apartments = [dict(row) for row in cursor.fetchall()]
    
    # Get documents
    cursor.execute("""
        SELECT d.doc_id, d.apt_id, d.title, d.version, d.status, d.created_at, a.name as apt_name
        FROM documents d
        LEFT JOIN apartments a ON d.apt_id = a.apt_id
        ORDER BY d.created_at DESC
    """)
    documents = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "apartments": apartments,
        "documents": documents,
        "active_tab": "upload"
    })


@router.get("/ui/quality", response_class=HTMLResponse)
def quality_page(request: Request):
    """Screen B: Quality Gate."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get documents for dropdown
    cursor.execute("""
        SELECT doc_id, title, version, status FROM documents 
        WHERE status != 'ARCHIVED' 
        ORDER BY created_at DESC
    """)
    documents = [dict(row) for row in cursor.fetchall()]
    
    # Get issues grouped by severity
    cursor.execute("""
        SELECT i.*, d.title as doc_title
        FROM qa_issues i
        LEFT JOIN documents d ON i.doc_id = d.doc_id
        ORDER BY 
            CASE i.severity WHEN 'RED' THEN 1 WHEN 'YELLOW' THEN 2 ELSE 3 END,
            i.issue_id
    """)
    issues = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return templates.TemplateResponse("quality.html", {
        "request": request,
        "documents": documents,
        "issues": issues,
        "active_tab": "quality"
    })


@router.get("/ui/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    """Screen C: Chat."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT apt_id, name FROM apartments ORDER BY name")
    apartments = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "apartments": apartments,
        "active_tab": "chat"
    })


@router.get("/ui/improvements", response_class=HTMLResponse)
def improvements_page(request: Request):
    """Screen D: Improvements + API Spec."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get apartments for dropdown
    cursor.execute("SELECT apt_id, name FROM apartments ORDER BY name")
    apartments = [dict(row) for row in cursor.fetchall()]
    
    # Get documents for dropdown
    cursor.execute("SELECT doc_id, title, version FROM documents WHERE status = 'APPROVED' ORDER BY created_at DESC")
    documents = [dict(row) for row in cursor.fetchall()]
    
    # Get improvement suggestions
    cursor.execute("""
        SELECT s.*, a.name as apt_name
        FROM improve_suggestions s
        LEFT JOIN apartments a ON s.apt_id = a.apt_id
        ORDER BY s.created_at DESC
    """)
    suggestions = [dict(row) for row in cursor.fetchall()]
    
    # Get API specs
    cursor.execute("""
        SELECT sp.*, d.title as doc_title
        FROM api_specs sp
        LEFT JOIN documents d ON sp.doc_id = d.doc_id
        ORDER BY sp.created_at DESC
    """)
    api_specs = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return templates.TemplateResponse("improvements.html", {
        "request": request,
        "apartments": apartments,
        "documents": documents,
        "suggestions": suggestions,
        "api_specs": api_specs,
        "active_tab": "improvements"
    })
