# Load environment variables FIRST, before any other imports that might check them
from dotenv import load_dotenv
load_dotenv()

import pytest
from unittest.mock import Mock, MagicMock


# =============================================================================
# Vector Store Fixtures
# =============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing tools without ChromaDB"""
    from vector_store import SearchResults
    store = Mock()
    store.search = Mock(return_value=SearchResults(
        documents=["Test content about AI"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    ))
    store.get_lesson_link = Mock(return_value="http://example.com/lesson1")
    store._resolve_course_name = Mock(return_value="Test Course")
    store.get_all_courses_metadata = Mock(return_value=[{
        "title": "Test Course",
        "course_link": "http://example.com",
        "lessons": [{"lesson_number": 1, "lesson_title": "Intro"}]
    }])
    return store


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing AI generator"""
    manager = Mock()
    manager.execute_tool = Mock(return_value="Search result: Test content")
    manager.get_last_sources = Mock(return_value=[])
    return manager


# =============================================================================
# Anthropic API Fixtures
# =============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls"""
    client = Mock()

    # Create a mock response structure
    mock_response = Mock()
    mock_response.content = [Mock(type="text", text="Test response")]
    mock_response.stop_reason = "end_turn"

    client.messages.create = Mock(return_value=mock_response)
    return client


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic response that triggers tool use"""
    mock_response = Mock()
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tool_123"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "test query"}

    mock_response.content = [mock_tool_block]
    mock_response.stop_reason = "tool_use"
    return mock_response


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    from config import Config
    config = Config()
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    return config


# =============================================================================
# Sample Test Data
# =============================================================================

@pytest.fixture
def sample_course_metadata():
    """Sample course metadata for testing"""
    return {
        "title": "Introduction to Machine Learning",
        "course_link": "http://example.com/ml-course",
        "instructor": "Dr. Test",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Welcome", "lesson_link": "http://example.com/ml-course/0"},
            {"lesson_number": 1, "lesson_title": "Basics of ML", "lesson_link": "http://example.com/ml-course/1"},
            {"lesson_number": 2, "lesson_title": "Neural Networks", "lesson_link": "http://example.com/ml-course/2"},
        ]
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    from vector_store import SearchResults
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks."
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 2},
            {"course_title": "Deep Learning Fundamentals", "lesson_number": 1}
        ],
        distances=[0.15, 0.22, 0.31]
    )


@pytest.fixture
def sample_query_request():
    """Sample query request for API testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123"
    }


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create a test client for FastAPI app"""
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for isolated API testing"""
    rag = Mock()
    rag.query = Mock(return_value=("Test answer", [{"source": "test"}]))
    rag.get_course_analytics = Mock(return_value={
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"]
    })
    rag.session_manager = Mock()
    rag.session_manager.create_session = Mock(return_value="new-session-id")
    rag.session_manager.clear_session = Mock()
    return rag
