"""
Integration tests for the full RAGSystem content-query path.

Uses the REAL ChromaDB (read-only) but a mocked Anthropic API so no
network calls or API keys are needed.  This layer catches failures that
only appear when all components are wired together:
  - VectorStore exceptions that slip past the try/except in search()
  - Malformed tool-result message structure
  - Sources leaking between queries
  - Pydantic serialisation problems in the response shape
"""

import os
import pytest
from unittest.mock import MagicMock, patch

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


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


class _FakeConfig:
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-test"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = CHROMA_PATH


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rag(request):
    """RAGSystem wired to real ChromaDB, Anthropic client mocked."""
    if not os.path.exists(CHROMA_PATH):
        pytest.skip("chroma_db not found – skipping RAG integration tests")

    with patch("ai_generator.anthropic.Anthropic"):
        from rag_system import RAGSystem

        system = RAGSystem(_FakeConfig())
    return system


# ── helpers on the fixture ────────────────────────────────────────────────────


def _prime(rag, *side_effects):
    """Reset call history and set side_effect on the mocked client."""
    rag.ai_generator.client.messages.create.reset_mock()
    rag.ai_generator.client.messages.create.side_effect = list(side_effects)
    rag.ai_generator.client.messages.create.return_value = None


def _prime_direct(rag, text: str):
    """Reset call history and prime for a single direct-text response."""
    rag.ai_generator.client.messages.create.reset_mock()
    rag.ai_generator.client.messages.create.side_effect = None
    rag.ai_generator.client.messages.create.return_value = _response(
        "end_turn", [_text_block(text)]
    )


# ── tests ─────────────────────────────────────────────────────────────────────


class TestDirectResponse:

    def test_direct_answer_returned_unchanged(self, rag):
        _prime_direct(rag, "This is a direct answer.")
        answer, sources = rag.query("What is 2+2?")
        assert answer == "This is a direct answer."
        assert sources == []


class TestContentQueryWithRealSearch:

    def test_search_tool_executes_and_results_are_synthesized(self, rag):
        """
        Simulate Claude choosing to call search_course_content.
        The real ChromaDB is queried; the mock second-call returns a fixed answer.
        """
        _prime(
            rag,
            _response(
                "tool_use",
                [_tool_use_block("tid_1", "search_course_content", {"query": "MCP server"})],
            ),
            _response("end_turn", [_text_block("MCP is a protocol for AI tools.")]),
        )
        answer, sources = rag.query("What is MCP?")
        assert answer == "MCP is a protocol for AI tools."
        assert isinstance(sources, list)

    def test_sources_are_populated_from_real_search(self, rag):
        """Sources returned from a tool-use query must be a list of dicts with label key."""
        _prime(
            rag,
            _response(
                "tool_use",
                [_tool_use_block("tid_1", "search_course_content", {"query": "MCP architecture"})],
            ),
            _response("end_turn", [_text_block("Answer about MCP.")]),
        )
        _, sources = rag.query("Describe MCP architecture")
        for s in sources:
            assert "label" in s, f"Source missing 'label': {s}"

    def test_empty_search_results_do_not_raise(self, rag):
        """A query that returns no search results must return a string, not raise."""
        _prime(
            rag,
            _response(
                "tool_use",
                [_tool_use_block("tid_1", "search_course_content", {"query": "xyzzy unknown"})],
            ),
            _response("end_turn", [_text_block("I found no relevant content.")]),
        )
        try:
            answer, sources = rag.query("Tell me about xyzzy")
            assert isinstance(answer, str)
        except Exception as exc:
            pytest.fail(f"query() raised unexpectedly: {exc}")


class TestMessageStructurePassedToApi:

    def test_second_api_call_has_three_messages(self, rag):
        """
        After tool execution the second API call must carry exactly three messages:
          [0] user original query
          [1] assistant tool_use block
          [2] user tool_result block
        A different count indicates a structural bug in _handle_tool_execution.
        """
        _prime(
            rag,
            _response(
                "tool_use",
                [_tool_use_block("tid_1", "search_course_content", {"query": "test topic"})],
            ),
            _response("end_turn", [_text_block("Answer")]),
        )
        rag.query("test topic query")

        second_call_kwargs = rag.ai_generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        assert len(messages) == 3, (
            f"Expected [user, assistant, user(tool_result)] but got {len(messages)} messages:\n"
            + "\n".join(f"  [{i}] role={m['role']}" for i, m in enumerate(messages))
        )

    def test_second_api_call_includes_tools(self, rag):
        """
        CRITICAL: the follow-up call must include `tools`.
        Without it the Anthropic API returns HTTP 400 for any request whose
        messages contain tool_use/tool_result blocks → uncaught exception
        → HTTP 500 → frontend 'query failed'.
        """
        _prime(
            rag,
            _response(
                "tool_use", [_tool_use_block("tid_1", "search_course_content", {"query": "test"})]
            ),
            _response("end_turn", [_text_block("ok")]),
        )
        rag.query("test query")

        second_call_kwargs = rag.ai_generator.client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs, (
            "BUG CONFIRMED at integration level: `tools` absent from second API call.\n"
            "Fix: in _handle_tool_execution() (ai_generator.py) add\n"
            "  'tools': base_params['tools']\n"
            "to final_params."
        )


class TestSourcesIsolation:

    def test_sources_from_tool_query_do_not_leak_into_next_query(self, rag):
        """Sources must be reset between queries."""
        # Query 1: tool use → populates sources
        _prime(
            rag,
            _response(
                "tool_use", [_tool_use_block("tid_1", "search_course_content", {"query": "MCP"})]
            ),
            _response("end_turn", [_text_block("MCP answer")]),
        )
        rag.query("What is MCP?")

        # Query 2: direct answer → sources must be empty
        _prime_direct(rag, "Hello!")
        _, sources_2 = rag.query("Say hello")
        assert (
            sources_2 == []
        ), f"Sources leaked from previous query into a direct-answer query: {sources_2}"
