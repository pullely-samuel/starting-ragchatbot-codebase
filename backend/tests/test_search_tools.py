from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Tests for CourseSearchTool.execute()"""

    def test_execute_returns_formatted_results(self, mock_vector_store):
        """Test that execute returns properly formatted results"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        assert "[Test Course - Lesson 1]" in result
        assert "Test content about AI" in result

    def test_execute_calls_vector_store_search(self, mock_vector_store):
        """Test that execute calls the vector store's search method"""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test query", course_name="Python", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Python", lesson_number=1
        )

    def test_execute_with_error_returns_error_message(self, mock_vector_store):
        """Test that errors from vector store are propagated"""
        mock_vector_store.search.return_value = SearchResults.empty(
            "Search error: Database connection failed"
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert "Search error" in result

    def test_execute_with_invalid_course_returns_error(self, mock_vector_store):
        """Test handling of invalid course name"""
        mock_vector_store.search.return_value = SearchResults.empty(
            "No course found matching 'invalid'"
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="invalid")

        assert "No course found" in result

    def test_execute_empty_results_returns_message(self, mock_vector_store):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, mock_vector_store):
        """Test empty results message includes filter info"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="Python", lesson_number=3)

        assert "Python" in result
        assert "lesson 3" in result

    def test_execute_tracks_sources(self, mock_vector_store):
        """Test that last_sources is populated after search"""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "http://example.com/lesson1"

    def test_execute_sources_without_lesson(self, mock_vector_store):
        """Test source tracking when lesson_number is None"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": None}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert tool.last_sources[0]["text"] == "Test Course"

    def test_get_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition has correct structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool.execute()"""

    def test_execute_returns_formatted_outline(self, mock_vector_store):
        """Test that execute returns properly formatted outline"""
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Test Course")

        assert "Course: Test Course" in result
        assert "Lessons:" in result

    def test_execute_invalid_course_returns_error(self, mock_vector_store):
        """Test handling of invalid course name"""
        mock_vector_store._resolve_course_name.return_value = None
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Nonexistent")

        assert "No course found" in result

    def test_get_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition has correct structure"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert definition["input_schema"]["required"] == ["course_name"]


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_execute_tool_calls_correct_tool(self, mock_vector_store):
        """Test that execute_tool calls the right tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")
        assert "[Test Course" in result

    def test_execute_unknown_tool_returns_error(self):
        """Test handling of unknown tool name"""
        manager = ToolManager()
        result = manager.execute_tool("unknown_tool")

        assert "not found" in result

    def test_get_tool_definitions_returns_all(self, mock_vector_store):
        """Test that get_tool_definitions returns all registered tools"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))
        manager.register_tool(CourseOutlineTool(mock_vector_store))

        definitions = manager.get_tool_definitions()
        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_get_last_sources(self, mock_vector_store):
        """Test retrieving sources from tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search then reset
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        sources = manager.get_last_sources()
        assert sources == []
