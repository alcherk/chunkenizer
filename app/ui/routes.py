"""UI route handlers."""
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import Optional

from app.db.database import get_db
from app.db.models import Document

router = APIRouter()
templates = Jinja2Templates(directory="app/ui/templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/documents/ui", response_class=HTMLResponse)
async def documents_ui(request: Request, db: Session = Depends(get_db)):
    """Documents list page."""
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return templates.TemplateResponse(
        "documents.html",
        {
            "request": request,
            "documents": [doc.to_dict() for doc in documents]
        }
    )


@router.get("/search/ui", response_class=HTMLResponse)
async def search_ui(request: Request):
    """Search page."""
    return templates.TemplateResponse("search.html", {"request": request})

