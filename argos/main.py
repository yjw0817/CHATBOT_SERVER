"""Argos Apartment AI - FastAPI Application."""
from pathlib import Path

# Load .env file before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import init_db
from routers import api, ui

# Initialize database on startup
init_db()

app = FastAPI(
    title="Argos Apartment AI",
    description="Sales-led RAG Builder Demo",
    version="0.1.0"
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(api.router)
app.include_router(ui.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
