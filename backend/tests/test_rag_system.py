from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.unit


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_returns_response_and_sources(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that query returns response and sources tuple"""
        from config import Config
        from rag_system import RAGSystem

        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Test answer"
        mock_ai_cls.return_value = mock_ai_instance

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)
        response, sources = system.query("test question")

        assert response == "Test answer"
        assert isinstance(sources, list)

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_calls_ai_generator_with_tools(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that query passes tools to AI generator"""
        from config import Config
        from rag_system import RAGSystem

        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_cls.return_value = mock_ai_instance

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)
        system.query("test")

        call_args = mock_ai_instance.generate_response.call_args
        assert "tools" in call_args.kwargs
        assert "tool_manager" in call_args.kwargs

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_includes_session_history(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that query retrieves and includes session history"""
        from config import Config
        from rag_system import RAGSystem

        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = "Previous chat"
        mock_session_cls.return_value = mock_session_instance

        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_cls.return_value = mock_ai_instance

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)
        system.query("test", session_id="session123")

        mock_session_instance.get_conversation_history.assert_called_with("session123")
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args.kwargs["conversation_history"] == "Previous chat"

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_updates_session_after_response(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that query updates session history after getting response"""
        from config import Config
        from rag_system import RAGSystem

        mock_session_instance = Mock()
        mock_session_cls.return_value = mock_session_instance

        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_cls.return_value = mock_ai_instance

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)
        system.query("test question", session_id="session123")

        mock_session_instance.add_exchange.assert_called_once()
        call_args = mock_session_instance.add_exchange.call_args
        assert call_args[0][0] == "session123"
        assert "test question" in call_args[0][1]
        assert call_args[0][2] == "Answer"


class TestRAGSystemToolRegistration:
    """Tests for tool registration in RAGSystem"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_registers_search_tool(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that CourseSearchTool is registered"""
        from config import Config
        from rag_system import RAGSystem

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)

        assert "search_course_content" in system.tool_manager.tools

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_registers_outline_tool(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that CourseOutlineTool is registered"""
        from config import Config
        from rag_system import RAGSystem

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)

        assert "get_course_outline" in system.tool_manager.tools


class TestRAGSystemSourceHandling:
    """Tests for source handling in RAGSystem"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_resets_sources_after_retrieval(
        self, mock_session_cls, mock_doc_cls, mock_vector_cls, mock_ai_cls
    ):
        """Test that sources are reset after being retrieved"""
        from config import Config
        from rag_system import RAGSystem

        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_cls.return_value = mock_ai_instance

        config = Config()
        config.MAX_RESULTS = 5

        system = RAGSystem(config)

        # Simulate sources being set
        system.search_tool.last_sources = [{"text": "Source"}]

        system.query("test")

        # Sources should be reset
        assert system.search_tool.last_sources == []
