"""
Integration tests that verify real API connectivity.

These tests are skipped if ANTHROPIC_API_KEY is not set.
Run with: uv run pytest backend/tests/test_integration.py -v
"""
import pytest
import os

# Skip all tests in this module if no API key available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration tests"
)


class TestAnthropicAPIIntegration:
    """Tests that verify the Anthropic API key and connection work"""

    def test_api_key_is_valid(self):
        """Test that the configured API key can make a real API call"""
        from ai_generator import AIGenerator
        from config import config

        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        response = generator.generate_response("Say 'hello' and nothing else")

        assert response is not None
        assert len(response) > 0

    def test_api_returns_text_response(self):
        """Test that the API returns a sensible text response"""
        from ai_generator import AIGenerator
        from config import config

        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        response = generator.generate_response("What is 2 + 2? Reply with just the number.")

        assert response is not None
        assert "4" in response
