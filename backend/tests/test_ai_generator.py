from unittest.mock import Mock, patch

from ai_generator import AIGenerator

pytestmark = pytest.mark.unit


class TestAIGeneratorBasic:
    """Tests for basic AIGenerator functionality"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_init_creates_client(self, mock_anthropic):
        """Test that initialization creates Anthropic client"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        mock_anthropic.assert_called_once_with(api_key="test-key")
        assert generator.model == "claude-sonnet-4-20250514"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_calls_api(self, mock_anthropic):
        """Test that generate_response makes API call"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("test query")

        assert result == "Test response"
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_system_prompt(self, mock_anthropic):
        """Test that system prompt is included in API call"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.generate_response("query")

        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "AI assistant" in call_args.kwargs["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_history(self, mock_anthropic):
        """Test that conversation history is included"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.generate_response("query", conversation_history="Previous messages")

        call_args = mock_client.messages.create.call_args
        assert "Previous messages" in call_args.kwargs["system"]


class TestAIGeneratorToolUse:
    """Tests for tool calling functionality"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_added_to_api_call(self, mock_anthropic):
        """Test that tools are included in API call when provided"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        tools = [{"name": "test_tool", "description": "A test tool"}]
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.generate_response("query", tools=tools)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_triggers_execution(self, mock_anthropic, mock_tool_manager):
        """Test that tool_use stop_reason triggers tool execution"""
        mock_client = Mock()

        # First response triggers tool use
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool_123"

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block]

        # Second response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test"
        )
        assert result == "Final answer"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_results_sent_to_api(self, mock_anthropic, mock_tool_manager):
        """Test that tool results are sent back to API"""
        mock_client = Mock()

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool_123"

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block]

        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.generate_response(
            "test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Check second API call includes tool result
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Should have: user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_123"

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tool_use_returns_direct_response(self, mock_anthropic):
        """Test that non-tool responses are returned directly"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        mock_tool_manager = Mock()

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Tool manager should not be called
        mock_tool_manager.execute_tool.assert_not_called()
        assert result == "Direct response"


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator configuration"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_base_params_set_correctly(self, mock_anthropic):
        """Test that base API parameters are set correctly"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800


class TestSequentialToolCalling:
    """Tests for sequential tool calling functionality (max 2 rounds)"""

    @staticmethod
    def _make_tool_block(name, query, tool_id):
        """Helper to create a mock tool block"""
        block = Mock()
        block.type = "tool_use"
        block.name = name
        block.input = {"query": query}
        block.id = tool_id
        return block

    @staticmethod
    def _make_tool_response(tool_block):
        """Helper to create a mock tool_use response"""
        response = Mock()
        response.stop_reason = "tool_use"
        response.content = [tool_block]
        return response

    @staticmethod
    def _make_text_response(text):
        """Helper to create a mock text response"""
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text=text)]
        return response

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls(self, mock_anthropic, mock_tool_manager):
        """Test that two sequential tool calls work with 3 API calls"""
        mock_client = Mock()

        # Round 1: tool call
        first_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "MCP", "tool_1")
        )

        # Round 2: another tool call
        second_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "lesson 3", "tool_2")
        )

        # Final: text response
        final_response = self._make_text_response("Final answer after two searches")

        mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Final answer after two searches"

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_enforced(self, mock_anthropic, mock_tool_manager):
        """Test that third API call has no tools (forces text response)"""
        mock_client = Mock()

        # Both rounds try to use tools
        first_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "test1", "tool_1")
        )
        second_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "test2", "tool_2")
        )

        # Final response when tools are removed
        final_response = self._make_text_response("Forced final answer")

        mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Third call should NOT have tools
        third_call_args = mock_client.messages.create.call_args_list[2]
        assert "tools" not in third_call_args.kwargs
        assert result == "Forced final answer"

    @patch("ai_generator.anthropic.Anthropic")
    def test_early_termination_no_tool_use(self, mock_anthropic, mock_tool_manager):
        """Test that loop exits when Claude doesn't use tools"""
        mock_client = Mock()

        # First round: tool call
        first_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "test", "tool_1")
        )

        # Second round: Claude decides not to use another tool
        second_response = self._make_text_response("Answer after one tool call")

        mock_client.messages.create.side_effect = [first_response, second_response]
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_tool_manager.execute_tool.call_count == 1
        assert mock_client.messages.create.call_count == 2
        assert result == "Answer after one tool call"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_passed_to_claude(self, mock_anthropic):
        """Test that tool errors are passed to Claude gracefully"""
        mock_client = Mock()

        first_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "test", "tool_1")
        )

        # Response after error is sent back to Claude
        error_response = self._make_text_response("I encountered an error searching")

        mock_client.messages.create.side_effect = [first_response, error_response]
        mock_anthropic.return_value = mock_client

        # Tool manager that throws an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection failed"
        )

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Should still get a response (Claude handles the error)
        assert result == "I encountered an error searching"
        # Verify error was sent in tool result
        second_call = mock_client.messages.create.call_args_list[1]
        tool_result = second_call.kwargs["messages"][2]["content"][0]
        assert tool_result["is_error"] is True
        assert "Database connection failed" in tool_result["content"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_messages_preserved_across_rounds(self, mock_anthropic, mock_tool_manager):
        """Test that message history accumulates correctly"""
        mock_client = Mock()

        first_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "first", "tool_1")
        )
        second_response = self._make_tool_response(
            self._make_tool_block("get_course_outline", "MCP", "tool_2")
        )
        final_response = self._make_text_response("Final")

        mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.generate_response(
            "query",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        # Third call should have full message history:
        # user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        final_call_args = mock_client.messages.create.call_args_list[2]
        messages = final_call_args.kwargs["messages"]
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_included_in_both_rounds(self, mock_anthropic, mock_tool_manager):
        """Test that tools are present in API calls for both rounds"""
        mock_client = Mock()

        first_response = self._make_tool_response(
            self._make_tool_block("search_course_content", "test", "tool_1")
        )
        second_response = self._make_text_response("Answer")

        mock_client.messages.create.side_effect = [first_response, second_response]
        mock_anthropic.return_value = mock_client

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.generate_response(
            "query", tools=tools, tool_manager=mock_tool_manager
        )

        # Both calls should include tools
        for call in mock_client.messages.create.call_args_list:
            assert "tools" in call.kwargs
            assert call.kwargs["tools"] == tools
