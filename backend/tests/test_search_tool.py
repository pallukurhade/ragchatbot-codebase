"""
Tests for CourseSearchTool.execute() — the content-search layer.

Unit tests use a mocked VectorStore so they run without ChromaDB.
Integration tests (class TestCourseSearchToolIntegration) hit the
real chroma_db that the running server uses.
"""
import os
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool
from vector_store import SearchResults


# ── helpers ───────────────────────────────────────────────────────────────────

def _store_with_results(docs, metas, lesson_url=None):
    store = MagicMock()
    store.search.return_value = SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )
    store.get_lesson_link.return_value = lesson_url
    return store


# ── unit tests ────────────────────────────────────────────────────────────────

class TestExecuteReturnsFormattedText:

    def test_course_title_and_content_appear_in_result(self):
        store = _store_with_results(
            docs=["Python is a high-level language."],
            metas=[{"course_title": "Python 101", "lesson_number": 1}],
        )
        result = CourseSearchTool(store).execute(query="what is python")
        assert "Python 101" in result
        assert "Python is a high-level language." in result
        assert "Lesson 1" in result

    def test_multiple_results_are_all_included(self):
        store = _store_with_results(
            docs=["Content A", "Content B"],
            metas=[
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course Y", "lesson_number": 2},
            ],
        )
        result = CourseSearchTool(store).execute(query="topic")
        assert "Content A" in result
        assert "Content B" in result

    def test_no_lesson_number_in_metadata_does_not_crash(self):
        store = _store_with_results(
            docs=["Intro text"],
            metas=[{"course_title": "Some Course"}],  # lesson_number key absent
        )
        result = CourseSearchTool(store).execute(query="intro")
        assert "Some Course" in result


class TestExecuteEmptyAndErrorCases:

    def test_empty_results_returns_no_content_message(self):
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        result = CourseSearchTool(store).execute(query="xyzzy nothing here")
        assert "No relevant content found" in result

    def test_empty_results_with_course_filter_mentions_course(self):
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        result = CourseSearchTool(store).execute(query="x", course_name="Nonexistent Course")
        assert "No relevant content found" in result
        assert "Nonexistent Course" in result

    def test_search_error_is_returned_as_string_not_raised(self):
        store = MagicMock()
        store.search.return_value = SearchResults.empty("Search error: ChromaDB failure")
        result = CourseSearchTool(store).execute(query="anything")
        assert "Search error" in result


class TestExecuteFiltersAreForwarded:

    def test_course_name_passed_to_vector_store(self):
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        CourseSearchTool(store).execute(query="python", course_name="Python 101")
        store.search.assert_called_once_with(
            query="python", course_name="Python 101", lesson_number=None
        )

    def test_lesson_number_passed_to_vector_store(self):
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        CourseSearchTool(store).execute(query="content", lesson_number=3)
        store.search.assert_called_once_with(
            query="content", course_name=None, lesson_number=3
        )

    def test_both_filters_passed_together(self):
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        CourseSearchTool(store).execute(query="q", course_name="MCP", lesson_number=2)
        store.search.assert_called_once_with(
            query="q", course_name="MCP", lesson_number=2
        )


class TestLastSources:

    def test_last_sources_populated_after_results(self):
        store = _store_with_results(
            docs=["Some content"],
            metas=[{"course_title": "Test Course", "lesson_number": 2}],
            lesson_url="http://example.com/lesson2",
        )
        tool = CourseSearchTool(store)
        tool.execute(query="test")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "Test Course - Lesson 2"
        assert tool.last_sources[0]["url"] == "http://example.com/lesson2"

    def test_last_sources_empty_when_no_results(self):
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)
        tool.execute(query="nothing")
        assert tool.last_sources == []

    def test_last_sources_url_is_none_when_no_lesson_link(self):
        store = _store_with_results(
            docs=["Content"],
            metas=[{"course_title": "Course A", "lesson_number": 1}],
            lesson_url=None,
        )
        tool = CourseSearchTool(store)
        tool.execute(query="q")
        assert tool.last_sources[0]["url"] is None


# ── integration tests (real ChromaDB) ────────────────────────────────────────

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


@pytest.fixture(scope="module")
def real_search_tool():
    if not os.path.exists(CHROMA_PATH):
        pytest.skip("chroma_db not found – skipping integration tests")
    from vector_store import VectorStore
    store = VectorStore(
        chroma_path=CHROMA_PATH,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,
    )
    return CourseSearchTool(store)


class TestCourseSearchToolIntegration:

    def test_search_returns_non_empty_result_for_known_topic(self, real_search_tool):
        result = real_search_tool.execute(query="MCP server")
        assert result, "Expected non-empty result"
        assert not result.startswith("Search error"), f"Search error: {result}"
        assert "No relevant content found" not in result

    def test_execute_never_raises(self, real_search_tool):
        """execute() must return a string for any input — never raise."""
        try:
            result = real_search_tool.execute(query="what is a chatbot")
            assert isinstance(result, str)
        except Exception as exc:
            pytest.fail(f"execute() raised unexpectedly: {exc}")

    def test_course_name_filter_works(self, real_search_tool):
        result = real_search_tool.execute(query="lesson content", course_name="MCP")
        assert isinstance(result, str)
        assert not result.startswith("Search error"), f"Filtered search error: {result}"

    def test_last_sources_populated_after_real_search(self, real_search_tool):
        real_search_tool.execute(query="MCP architecture")
        # last_sources may be empty if search returns nothing, but must be a list
        assert isinstance(real_search_tool.last_sources, list)
