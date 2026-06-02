# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh

# Run manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App runs at `http://localhost:8000`. Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=...` (see `.env.example`).

**Always use `uv` to run the server. Never use `pip` or invoke Python directly.**

## Architecture

This is a **RAG (Retrieval-Augmented Generation) chatbot** backed by FastAPI. The backend serves the frontend as static files — there is no separate frontend dev server.

### Request flow

1. `frontend/script.js` posts `{ query, session_id }` to `POST /api/query`
2. `backend/app.py` routes to `RAGSystem.query()`
3. `rag_system.py` fetches conversation history and calls `AIGenerator.generate_response()` with the `search_course_content` tool available
4. Claude decides whether to invoke the tool. If it does, `search_tools.py` runs a semantic search against ChromaDB and returns formatted chunks
5. Claude makes a second API call to synthesize tool results into a final answer
6. Sources and answer are returned to the frontend

### Key design decisions

**Tool-based retrieval**: Rather than injecting retrieved chunks directly into the prompt, the system exposes a `search_course_content` tool to Claude (via Anthropic's tool use API). Claude decides when and how to search. This means general knowledge questions bypass the vector store entirely.

**Two ChromaDB collections**: `course_catalog` stores per-course metadata and is used for fuzzy course name resolution (semantic lookup). `course_content` stores text chunks and is queried for actual content retrieval. Course title is used as the document ID in the catalog.

**Session history as plain text**: Conversation history is formatted as a string appended to the system prompt, not passed as multi-turn `messages`. This is in `session_manager.py` — history is capped at the last 2 exchanges (4 messages).

**Deduplication on startup**: On startup, `app.py` calls `add_course_folder()` which checks `get_existing_course_titles()` before indexing, so restarting the server does not re-embed already-loaded courses.

### Document format

Course files in `docs/` must follow this structure for `document_processor.py` to parse them correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <title>
...
```

Lessons are chunked at 800 characters with 100-character sentence-level overlap. The first chunk of each lesson is prefixed with `"Lesson N content: ..."` to preserve lesson context in the embedding.

### Configuration

All tunable parameters are in `backend/config.py`: model name, chunk size/overlap, max search results, history length, and ChromaDB path.
