"""
Tests for AIGenerator — verifies the tool-use call flow with a mocked Anthropic client.

The critical test is test_second_api_call_includes_tools_parameter:
  The Anthropic API requires `tools` to be present in ANY call whose `messages`
  contain tool_use or tool_result content blocks.  The current implementation
  builds final_params WITHOUT `tools`, which causes the API to return a 400
  Bad Request — propagating as an uncaught exception → HTTP 500 → "query failed".
"""
import pytest
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


# ── helpers ───────────────────────────────────────────────────────────────────

def _text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(tool_id: str, name: str, input_data: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_data
    return b


def _response(stop_reason: str, blocks: list):
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = blocks
    return r


FAKE_TOOLS = [{"name": "search_course_content", "description": "search", "input_schema": {}}]


@pytest.fixture
def gen():
    """AIGenerator with a fully mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic"):
        g = AIGenerator(api_key="fake-key", model="claude-test")
    return g


# ── direct-text response (no tool use) ───────────────────────────────────────

class TestDirectTextResponse:

    def test_returns_text_block_content(self, gen):
        gen.client.messages.create.return_value = _response(
            "end_turn", [_text_block("Direct answer.")]
        )
        assert gen.generate_response(query="What is Python?") == "Direct answer."

    def test_makes_exactly_one_api_call(self, gen):
        gen.client.messages.create.return_value = _response("end_turn", [_text_block("ok")])
        gen.generate_response(query="Hello")
        assert gen.client.messages.create.call_count == 1

    def test_returns_empty_string_when_response_has_no_text_block(self, gen):
        gen.client.messages.create.return_value = _response("end_turn", [])
        assert gen.generate_response(query="Hello") == ""


# ── tool-use call flow ────────────────────────────────────────────────────────

class TestToolUseFlow:

    def _two_call_setup(self, gen, tool_input=None):
        if tool_input is None:
            tool_input = {"query": "python basics"}
        first = _response("tool_use", [
            _tool_use_block("tid_1", "search_course_content", tool_input)
        ])
        second = _response("end_turn", [_text_block("Python is a language.")])
        gen.client.messages.create.side_effect = [first, second]
        tm = MagicMock()
        tm.execute_tool.return_value = "Found: Python basics content"
        return tm

    def test_tool_manager_execute_tool_is_called(self, gen):
        tm = self._two_call_setup(gen, {"query": "python"})
        gen.generate_response(query="Tell me about Python", tools=FAKE_TOOLS, tool_manager=tm)
        tm.execute_tool.assert_called_once_with("search_course_content", query="python")

    def test_two_api_calls_are_made(self, gen):
        tm = self._two_call_setup(gen)
        gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        assert gen.client.messages.create.call_count == 2

    def test_final_text_is_returned(self, gen):
        tm = self._two_call_setup(gen)
        result = gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        assert result == "Python is a language."

    def test_second_call_messages_have_three_turns(self, gen):
        """Messages for the second call must be: [user, assistant(tool_use), user(tool_result)]."""
        tm = self._two_call_setup(gen, {"query": "lesson content"})
        gen.generate_response(query="What is in lesson 1?", tools=FAKE_TOOLS, tool_manager=tm)
        kwargs = gen.client.messages.create.call_args_list[1][1]
        messages = kwargs["messages"]
        assert len(messages) == 3, (
            f"Expected 3 messages but got {len(messages)}: {messages}"
        )

    def test_second_call_last_message_is_tool_result(self, gen):
        tm = self._two_call_setup(gen, {"query": "lesson content"})
        tm.execute_tool.return_value = "Lesson 1: Introduction to Python"
        gen.generate_response(query="What is in lesson 1?", tools=FAKE_TOOLS, tool_manager=tm)
        kwargs = gen.client.messages.create.call_args_list[1][1]
        last_msg = kwargs["messages"][-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["content"] == "Lesson 1: Introduction to Python"

    def test_second_call_assistant_message_has_tool_use_block(self, gen):
        tm = self._two_call_setup(gen)
        gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        kwargs = gen.client.messages.create.call_args_list[1][1]
        assistant_msg = kwargs["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in assistant_msg["content"])

    def test_second_api_call_includes_tools_parameter(self, gen):
        """
        CRITICAL BUG TEST.

        The Anthropic API requires `tools` to be present in any request whose
        `messages` contain tool_use or tool_result blocks.  Without it the API
        returns HTTP 400, which propagates as an uncaught exception all the way
        to app.py → HTTP 500 → the frontend shows "query failed".

        Current implementation builds final_params WITHOUT tools:
            final_params = {
                **self.base_params,     # model, temperature, max_tokens only
                "messages": messages,
                "system": base_params["system"]
            }

        Fix: add  tools=base_params["tools"]  to final_params inside
        _handle_tool_execution() in backend/ai_generator.py.
        """
        tm = self._two_call_setup(gen)
        gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]

        assert "tools" in second_call_kwargs, (
            "BUG: `tools` is missing from the second Anthropic API call.\n"
            "The API rejects requests that contain tool_use/tool_result blocks "
            "but have no `tools` definition.  This causes an uncaught exception "
            "→ HTTP 500 → frontend shows 'query failed'.\n"
            "Fix: in _handle_tool_execution() change:\n"
            "  final_params = {**self.base_params, 'messages': messages, 'system': ...}\n"
            "to:\n"
            "  final_params = {**self.base_params, 'messages': messages, 'system': ...,\n"
            "                  'tools': base_params['tools']}"
        )


# ── sequential tool calls (two rounds) ───────────────────────────────────────

class TestSequentialToolCalls:

    def _three_call_setup(self, gen, tool1_input=None, tool2_input=None):
        """round1=search, round2=outline, final=text — two distinct tool calls."""
        first  = _response("tool_use", [
            _tool_use_block("t1", "search_course_content",
                            tool1_input or {"query": "MCP"})
        ])
        second = _response("tool_use", [
            _tool_use_block("t2", "get_course_outline",
                            tool2_input or {"course_name": "MCP"})
        ])
        third  = _response("end_turn", [_text_block("Final synthesized answer.")])
        gen.client.messages.create.side_effect = [first, second, third]
        tm = MagicMock()
        tm.execute_tool.side_effect = ["Search results", "Course outline"]
        return tm

    def test_two_rounds_make_three_api_calls(self, gen):
        tm = self._three_call_setup(gen)
        result = gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        assert gen.client.messages.create.call_count == 3
        assert result == "Final synthesized answer."

    def test_both_tools_executed_in_order(self, gen):
        tm = self._three_call_setup(gen)
        gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        assert tm.execute_tool.call_count == 2
        assert tm.execute_tool.call_args_list[0][0][0] == "search_course_content"
        assert tm.execute_tool.call_args_list[1][0][0] == "get_course_outline"

    def test_third_api_call_has_five_messages(self, gen):
        """[user, asst(tool1), user(result1), asst(tool2), user(result2)]"""
        tm = self._three_call_setup(gen)
        gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        third_call_kwargs = gen.client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        assert len(messages) == 5, (
            f"Expected 5 messages in third call but got {len(messages)}"
        )

    def test_third_api_call_includes_tools(self, gen):
        tm = self._three_call_setup(gen)
        gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        third_call_kwargs = gen.client.messages.create.call_args_list[2][1]
        assert "tools" in third_call_kwargs

    def test_stops_after_first_round_if_text_returned(self, gen):
        """If round 1's follow-up returns text, no second tool round runs."""
        first  = _response("tool_use", [_tool_use_block("t1", "search_course_content", {"query": "x"})])
        second = _response("end_turn", [_text_block("One-round answer.")])
        gen.client.messages.create.side_effect = [first, second]
        tm = MagicMock()
        tm.execute_tool.return_value = "results"

        result = gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)

        assert gen.client.messages.create.call_count == 2
        assert result == "One-round answer."

    def test_round_2_tool_failure_falls_back_gracefully(self, gen):
        """If the second round's tool call raises, the fallback fires without re-raising."""
        first    = _response("tool_use", [_tool_use_block("t1", "search_course_content", {"query": "q"})])
        second   = _response("tool_use", [_tool_use_block("t2", "get_course_outline", {"course_name": "X"})])
        fallback = _response("end_turn", [_text_block("Fallback after failure.")])
        gen.client.messages.create.side_effect = [first, second, fallback]

        tm = MagicMock()
        tm.execute_tool.side_effect = ["Round1 result", RuntimeError("tool broken")]

        result = gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)

        assert result == "Fallback after failure."
        assert gen.client.messages.create.call_count == 3


# ── fallback path ─────────────────────────────────────────────────────────────

class TestFallbackPath:

    def test_three_calls_made_when_final_response_has_no_text(self, gen):
        first = _response("tool_use", [
            _tool_use_block("tid_1", "search_course_content", {"query": "test"})
        ])
        no_text = _response("end_turn", [])
        fallback = _response("end_turn", [_text_block("Fallback answer")])
        gen.client.messages.create.side_effect = [first, no_text, fallback]
        tm = MagicMock()
        tm.execute_tool.return_value = "Content"

        result = gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)

        assert gen.client.messages.create.call_count == 3
        assert result == "Fallback answer"

    def test_returns_empty_string_when_all_three_calls_have_no_text(self, gen):
        first = _response("tool_use", [
            _tool_use_block("tid_1", "search_course_content", {"query": "test"})
        ])
        empty = _response("end_turn", [])
        gen.client.messages.create.side_effect = [first, empty, empty]
        tm = MagicMock()
        tm.execute_tool.return_value = "Content"

        result = gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
        assert result == ""


# ── exception propagation ─────────────────────────────────────────────────────

class TestExceptionPropagation:

    def test_exception_in_first_call_propagates(self, gen):
        gen.client.messages.create.side_effect = RuntimeError("API unreachable")
        with pytest.raises(RuntimeError, match="API unreachable"):
            gen.generate_response(query="test")

    def test_exception_in_second_call_propagates_to_caller(self, gen):
        """
        An exception in the second (follow-up) API call is NOT caught inside
        _handle_tool_execution.  It travels up the stack:
          _handle_tool_execution → generate_response → rag_system.query()
          → app.py except block → HTTP 500 → frontend "query failed".
        This test documents that propagation so the fix is clear.
        """
        first = _response("tool_use", [
            _tool_use_block("tid_1", "search_course_content", {"query": "test"})
        ])
        gen.client.messages.create.side_effect = [
            first,
            RuntimeError("400 Bad Request: tools must be provided"),
        ]
        tm = MagicMock()
        tm.execute_tool.return_value = "Some content"

        with pytest.raises(RuntimeError, match="400 Bad Request"):
            gen.generate_response(query="test", tools=FAKE_TOOLS, tool_manager=tm)
