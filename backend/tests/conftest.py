import sys
import os

# Make backend/ importable from backend/tests/ when running pytest directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ── Pydantic models (mirrors app.py) ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SourceItem(BaseModel):
    label: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str

class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ── Test app factory ──────────────────────────────────────────────────────────

def make_test_app(rag_system) -> FastAPI:
    """Build a minimal FastAPI app with the same API routes but no static-file mount."""
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        rag_system.session_manager.clear_session(session_id)
        return Response(status_code=204)

    return app


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag():
    """Fully mocked RAGSystem with sensible defaults."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.query.return_value = ("Test answer.", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag


@pytest.fixture
def client(mock_rag):
    """TestClient for the minimal test app backed by mock_rag."""
    app = make_test_app(mock_rag)
    return TestClient(app)
