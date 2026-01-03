"""
End-to-end tests that verify the full HTTP API flow.

These tests use FastAPI's TestClient to make real HTTP requests.
Query tests are skipped if ANTHROPIC_API_KEY is not set.
Run with: uv run pytest backend/tests/test_e2e.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    """Create test client for the FastAPI app"""
    return TestClient(app)


@pytest.mark.e2e
class TestCoursesEndpoint:
    """Tests for endpoints that don't require the API key"""

    def test_courses_endpoint_returns_stats(self, client):  # Uses module-level fixture
        """Test courses stats endpoint returns valid structure"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping query E2E tests",
)
class TestQueryEndpointE2E:
    """End-to-end tests for the query endpoint (requires API key)"""

    def test_query_returns_valid_response(self, client):  # Uses module-level fixture
        """Test full query flow through HTTP endpoint"""
        response = client.post(
            "/api/query", json={"query": "What courses are available?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert len(data["answer"]) > 0

    def test_query_with_session_maintains_context(self, client):
        """Test that session ID allows conversation continuity"""
        # First query
        response1 = client.post("/api/query", json={"query": "Hello"})
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # Second query with same session
        response2 = client.post(
            "/api/query",
            json={"query": "What did I just say?", "session_id": session_id},
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

    def test_query_invalid_request_returns_error(self, client):
        """Test that invalid requests are handled properly"""
        response = client.post("/api/query", json={})

        # Should return 422 for validation error (missing required field)
        assert response.status_code == 422


@pytest.mark.e2e
class TestSessionEndpoint:
    """Tests for session management endpoint"""

    def test_clear_session_returns_success(self, client):  # Uses module-level fixture
        """Test clearing a session works"""
        response = client.delete("/api/session/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "test-session-123"
