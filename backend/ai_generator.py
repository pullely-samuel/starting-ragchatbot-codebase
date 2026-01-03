import anthropic
from typing import List, Optional, Dict

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Selection:
- Use `get_course_outline` for questions about course structure, syllabus, lesson lists, or what topics a course covers. When returning an outline, include the course title, course link, and for each lesson include both the lesson number and title.
- Use `search_course_content` for questions about specific content or topics within lessons

Tool Usage Rules:
- **Up to 2 tool calls per query**: You may make sequential tool calls if needed (e.g., get course outline first, then search specific content)
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling up to 2 rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages
        messages = [{"role": "user", "content": query}]

        # Tool calling loop (max 2 rounds)
        max_rounds = 2
        for round_num in range(max_rounds):
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Include tools for potential tool calls
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use" and tool_manager:
                tool_results = self._execute_tools(response, tool_manager)

                if tool_results is None:
                    return self._extract_text_response(response)

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                return self._extract_text_response(response)

        # Max rounds reached - force final response without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        final_response = self.client.messages.create(**final_params)
        return self._extract_text_response(final_response)

    def _execute_tools(self, response, tool_manager) -> Optional[List[Dict]]:
        """
        Execute all tool calls from response.

        Args:
            response: API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dicts, or None if no tools executed
        """
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {str(e)}",
                        "is_error": True
                    })
        return tool_results if tool_results else None

    def _extract_text_response(self, response) -> str:
        """
        Extract text from response content blocks.

        Args:
            response: API response object

        Returns:
            Text content as string, or empty string if no text found
        """
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""