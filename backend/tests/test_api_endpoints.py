"""
Tests for the FastAPI API endpoints.

Uses a minimal test app (from conftest.py) that mirrors app.py's routes
without the static-file mount, which fails when ../frontend does not exist.
"""
import pytest


# ── POST /api/query ───────────────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_returns_answer_and_session_id(self, client, mock_rag):
        mock_rag.query.return_value = ("Hello, world.", [])
        resp = client.post("/api/query", json={"query": "What is Python?", "session_id": "s1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Hello, world."
        assert data["session_id"] == "s1"
        assert data["sources"] == []

    def test_creates_new_session_when_none_provided(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer.", [])
        resp = client.post("/api/query", json={"query": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test-session-id"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id_without_creating_new(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer.", [])
        client.post("/api/query", json={"query": "Hello", "session_id": "existing-session"})
        mock_rag.session_manager.create_session.assert_not_called()

    def test_sources_returned_as_list_of_objects(self, client, mock_rag):
        mock_rag.query.return_value = (
            "Answer with source.",
            [{"label": "Course A - Lesson 1", "url": "http://example.com/l1"}],
        )
        resp = client.post("/api/query", json={"query": "Tell me about course A", "session_id": "s1"})
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["label"] == "Course A - Lesson 1"
        assert sources[0]["url"] == "http://example.com/l1"

    def test_source_url_may_be_null(self, client, mock_rag):
        mock_rag.query.return_value = (
            "Answer.",
            [{"label": "Course B - Lesson 2", "url": None}],
        )
        resp = client.post("/api/query", json={"query": "q", "session_id": "s1"})
        assert resp.status_code == 200
        assert resp.json()["sources"][0]["url"] is None

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_malformed_json_returns_422(self, client):
        resp = client.post("/api/query", content=b"not-json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_rag_exception_returns_500(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("AI service unavailable")
        resp = client.post("/api/query", json={"query": "test", "session_id": "s1"})
        assert resp.status_code == 500
        assert "AI service unavailable" in resp.json()["detail"]

    def test_rag_is_called_with_correct_args(self, client, mock_rag):
        mock_rag.query.return_value = ("ok", [])
        client.post("/api/query", json={"query": "my question", "session_id": "abc"})
        mock_rag.query.assert_called_once_with("my question", "abc")


# ── GET /api/courses ──────────────────────────────────────────────────────────

class TestCoursesEndpoint:

    def test_returns_total_courses_and_titles(self, client, mock_rag):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course A", "Course B"]

    def test_returns_empty_list_when_no_courses(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_exception_returns_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB connection failed")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "DB connection failed" in resp.json()["detail"]

    def test_course_count_matches_titles_length(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["X", "Y", "Z"],
        }
        resp = client.get("/api/courses")
        data = resp.json()
        assert data["total_courses"] == len(data["course_titles"])


# ── DELETE /api/session/{session_id} ─────────────────────────────────────────

class TestDeleteSessionEndpoint:

    def test_returns_204_no_content(self, client):
        resp = client.delete("/api/session/my-session")
        assert resp.status_code == 204

    def test_response_body_is_empty(self, client):
        resp = client.delete("/api/session/any-id")
        assert resp.content == b""

    def test_clears_correct_session(self, client, mock_rag):
        client.delete("/api/session/session-xyz")
        mock_rag.session_manager.clear_session.assert_called_once_with("session-xyz")

    def test_different_session_ids_are_forwarded(self, client, mock_rag):
        client.delete("/api/session/alpha")
        client.delete("/api/session/beta")
        calls = [c.args[0] for c in mock_rag.session_manager.clear_session.call_args_list]
        assert calls == ["alpha", "beta"]
