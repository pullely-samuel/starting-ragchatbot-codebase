# Load environment variables FIRST, before any other imports that might check them
from dotenv import load_dotenv

load_dotenv()

from unittest.mock import Mock  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing tools without ChromaDB"""
    from vector_store import SearchResults

    store = Mock()
    store.search = Mock(
        return_value=SearchResults(
            documents=["Test content about AI"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )
    )
    store.get_lesson_link = Mock(return_value="http://example.com/lesson1")
    store._resolve_course_name = Mock(return_value="Test Course")
    store.get_all_courses_metadata = Mock(
        return_value=[
            {
                "title": "Test Course",
                "course_link": "http://example.com",
                "lessons": [{"lesson_number": 1, "lesson_title": "Intro"}],
            }
        ]
    )
    return store


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing AI generator"""
    manager = Mock()
    manager.execute_tool = Mock(return_value="Search result: Test content")
    manager.get_last_sources = Mock(return_value=[])
    return manager
