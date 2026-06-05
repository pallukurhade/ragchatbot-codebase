import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course content and retrieving course outlines.

Tool Usage:
- **Outline / syllabus / lesson-list questions**: call `get_course_outline` — always include the course title, course link, and every lesson number with its title in your response
- **Content / detail questions**: call `search_course_content`
- **Up to two sequential tool calls per query**: if the first result is needed to inform a follow-up search, make a second tool call. Synthesize all results in your final answer.
- Once you have made two tool calls, provide your final text answer immediately.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without calling a tool
- **Course-specific questions**: Call the appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool-call explanations, or question-type analysis
 - Do not mention "based on the search results"

For course outline responses always present:
1. Course title (as a heading)
2. Course link
3. Numbered list of every lesson: "Lesson <number>: <title>"

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
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 2048}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, api_params, tool_manager)

        # Return direct response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _run_tool_loop(self, initial_response, base_params: Dict[str, Any], tool_manager) -> str:
        """
        Execute up to MAX_TOOL_ROUNDS sequential tool calls, building conversation
        context across rounds, then return the final synthesized text.
        """
        messages = base_params["messages"].copy()
        all_tool_results = []  # accumulated across rounds for the fallback
        current_response = initial_response
        rounds_used = 0

        while rounds_used < self.MAX_TOOL_ROUNDS:
            # Reconstruct assistant content — strip SDK-private fields so the API
            # doesn't silently drop the assistant turn when it's passed back.
            assistant_content = []
            for block in current_response.content:
                if block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute every tool call in this round; bail on the first failure
            tool_results = []
            failed = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception:
                        failed = True
                        break
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            all_tool_results.extend(tool_results)

            if failed or not tool_results:
                break

            messages.append({"role": "user", "content": tool_results})
            rounds_used += 1

            # Next API call — tools must always be re-sent when messages contain
            # tool_use / tool_result blocks (Anthropic API requirement).
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }
            if "tools" in base_params:
                next_params["tools"] = base_params["tools"]
                next_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**next_params)

            # Return immediately if Claude synthesized a text answer
            for block in current_response.content:
                if block.type == "text":
                    return block.text

            # Exit loop if Claude is done using tools
            if current_response.stop_reason != "tool_use":
                break
            # Otherwise loop again (rounds_used checked at the top)

        # One last text check on whatever the final response was
        for block in current_response.content:
            if block.type == "text":
                return block.text

        # Fallback: the tool-turn pattern was dropped by the API. Retry by
        # embedding all collected results directly in a plain user message.
        retrieved_text = "\n\n".join(tr["content"] for tr in all_tool_results)
        original_query = base_params["messages"][0]["content"]
        fallback_messages = [
            {
                "role": "user",
                "content": f"{original_query}\n\nRelevant course content:\n{retrieved_text}",
            }
        ]
        fallback_response = self.client.messages.create(
            **self.base_params, messages=fallback_messages, system=base_params["system"]
        )
        for block in fallback_response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
