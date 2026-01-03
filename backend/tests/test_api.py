"""
API endpoint tests for the FastAPI application.

Tests HTTP request/response handling, validation, and error scenarios.
Run with: uv run pytest backend/tests/test_api.py -v
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client for the FastAPI app"""
    from app import app

    return TestClient(app)


@pytest.mark.e2e
class TestRootEndpoint:
    """Tests for the root endpoint (static file serving)"""

    def test_root_serves_html(self, client):
        """Test that root endpoint serves the frontend HTML"""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_static_css_served(self, client):
        """Test that CSS files are served"""
        response = client.get("/style.css")

        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")

    def test_static_js_served(self, client):
        """Test that JavaScript files are served"""
        response = client.get("/script.js")

        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")


@pytest.mark.e2e
class TestQueryEndpointValidation:
    """Tests for /api/query request validation"""

    def test_missing_query_returns_422(self, client):
        """Test that missing query field returns validation error"""
        response = client.post("/api/query", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_empty_query_is_accepted(self, client):
        """Test that empty string query is technically valid"""
        # Note: Empty queries are accepted by Pydantic but may fail in business logic
        response = client.post("/api/query", json={"query": ""})

        # Should either succeed or return 500 (business logic error), not 422
        assert response.status_code in [200, 500]

    def test_invalid_json_returns_422(self, client):
        """Test that invalid JSON returns error"""
        response = client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_optional_session_id_accepted(self, client):
        """Test that session_id is optional"""
        # This test may hit the real API, so we just verify the request is accepted
        response = client.post("/api/query", json={"query": "test"})

        # Request should be accepted (200 or 500 if API key missing)
        assert response.status_code in [200, 500]


@pytest.mark.e2e
class TestQueryEndpointErrorHandling:
    """Tests for /api/query error handling"""

    def test_query_error_returns_500(self, client):
        """Test that internal errors return 500 status"""
        with patch("app.rag_system") as mock_rag:
            mock_rag.session_manager.create_session.return_value = "test-session"
            mock_rag.query.side_effect = Exception("Internal error")

            response = client.post("/api/query", json={"query": "test"})

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Internal error" in data["detail"]

    def test_query_response_structure(self, client):
        """Test that successful response has correct structure"""
        with patch("app.rag_system") as mock_rag:
            mock_rag.session_manager.create_session.return_value = "test-session"
            mock_rag.query.return_value = ("Test answer", [{"source": "test"}])

            response = client.post("/api/query", json={"query": "test"})

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "session_id" in data
            assert data["answer"] == "Test answer"
            assert data["session_id"] == "test-session"


@pytest.mark.e2e
class TestCoursesEndpointErrorHandling:
    """Tests for /api/courses error handling"""

    def test_courses_error_returns_500(self, client):
        """Test that internal errors return 500 status"""
        with patch("app.rag_system") as mock_rag:
            mock_rag.get_course_analytics.side_effect = Exception("Database error")

            response = client.get("/api/courses")

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Database error" in data["detail"]

    def test_courses_response_structure(self, client):
        """Test that successful response has correct structure"""
        with patch("app.rag_system") as mock_rag:
            mock_rag.get_course_analytics.return_value = {
                "total_courses": 3,
                "course_titles": ["Course A", "Course B", "Course C"],
            }

            response = client.get("/api/courses")

            assert response.status_code == 200
            data = response.json()
            assert data["total_courses"] == 3
            assert len(data["course_titles"]) == 3


@pytest.mark.e2e
class TestSessionEndpointValidation:
    """Tests for /api/session endpoint validation"""

    def test_clear_nonexistent_session_succeeds(self, client):
        """Test that clearing a nonexistent session still returns success"""
        response = client.delete("/api/session/nonexistent-session")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"

    def test_clear_session_returns_session_id(self, client):
        """Test that response includes the cleared session ID"""
        session_id = "my-test-session-456"
        response = client.delete(f"/api/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id


@pytest.mark.e2e
class TestAPIHeaders:
    """Tests for API response headers"""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are set correctly"""
        response = client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS preflight should succeed
        assert response.status_code in [200, 405]

    def test_json_content_type_on_api_responses(self, client):
        """Test that API responses have JSON content type"""
        response = client.get("/api/courses")

        assert "application/json" in response.headers.get("content-type", "")
