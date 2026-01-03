import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator


class TestAIGeneratorBasic:
    """Tests for basic AIGenerator functionality"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_init_creates_client(self, mock_anthropic):
        """Test that initialization creates Anthropic client"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        mock_anthropic.assert_called_once_with(api_key="test-key")
        assert generator.model == "claude-sonnet-4-20250514"

    @patch('ai_generator.anthropic.Anthropic')
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

    @patch('ai_generator.anthropic.Anthropic')
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

    @patch('ai_generator.anthropic.Anthropic')
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

    @patch('ai_generator.anthropic.Anthropic')
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

    @patch('ai_generator.anthropic.Anthropic')
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
            tool_manager=mock_tool_manager
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test"
        )
        assert result == "Final answer"

    @patch('ai_generator.anthropic.Anthropic')
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
            tool_manager=mock_tool_manager
        )

        # Check second API call includes tool result
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Should have: user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_123"

    @patch('ai_generator.anthropic.Anthropic')
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
            tool_manager=mock_tool_manager
        )

        # Tool manager should not be called
        mock_tool_manager.execute_tool.assert_not_called()
        assert result == "Direct response"


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator configuration"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_base_params_set_correctly(self, mock_anthropic):
        """Test that base API parameters are set correctly"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
