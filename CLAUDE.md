# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Users ask questions via a web interface, and Claude answers using semantically-searched course content from ChromaDB.

## Commands

```bash
# Run the application (from project root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Web interface: http://localhost:8000
API docs: http://localhost:8000/docs

**Always use `uv run` to execute Python scripts, not `python` directly.** This ensures the correct virtual environment and dependencies are used.

## Testing

```bash
# Run all tests
uv run pytest -v
```

Tests are located in `backend/tests/`. Run tests after major changes to verify nothing is broken. When modifying functionality, update corresponding tests to reflect new logic.

## Architecture

### Request Flow

```
Frontend (script.js) → POST /api/query
    → FastAPI (app.py)
    → RAGSystem.query() orchestrates:
        1. SessionManager: retrieves conversation history
        2. AIGenerator: calls Claude API with search tool
        3. If Claude uses tool → CourseSearchTool.execute()
            → VectorStore.search() (ChromaDB semantic search)
        4. Claude generates response from search results
    → Response with sources returned to frontend
```

### Key Components

| File | Purpose |
|------|---------|
| `backend/rag_system.py` | Main orchestrator - coordinates all components |
| `backend/ai_generator.py` | Claude API wrapper with tool execution loop |
| `backend/search_tools.py` | Tool definitions for Claude; `CourseSearchTool` performs vector search |
| `backend/vector_store.py` | ChromaDB interface with two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `backend/document_processor.py` | Parses course docs, extracts metadata, chunks text with overlap |
| `backend/session_manager.py` | Maintains conversation history per session |
| `backend/config.py` | Central configuration (chunk size, max results, model settings) |

### Tool-Calling Architecture

The system uses Claude's tool-calling capability rather than automatic RAG retrieval:
1. User query is sent to Claude with `search_course_content` tool definition
2. Claude decides whether to search based on the query
3. If `stop_reason == "tool_use"`, `AIGenerator._handle_tool_execution()` runs the tool
4. Tool results are sent back to Claude for final response generation

### Vector Store Collections

ChromaDB maintains two separate collections:
- **`course_catalog`**: Stores course metadata (title, instructor, lesson list) for course name resolution
- **`course_content`**: Stores chunked course content for semantic search with metadata filters (course_title, lesson_number)

### Document Processing

Course documents in `docs/` follow this format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content...]
```

Documents are chunked (800 chars, 100 overlap) and embedded on server startup.

### Frontend

Vanilla HTML/CSS/JS with no build step. Uses marked.js from CDN for markdown rendering.

#### Cache Busting

Static assets use version query parameters for cache busting:
- `style.css?v=12` in `frontend/index.html`
- `script.js?v=11` in `frontend/index.html`

**When modifying CSS or JS files**, increment the version number in `index.html` to ensure browsers load the updated files.

#### Playwright Browser Cache Bypass

When testing with Playwright MCP server, the browser may cache static files. To force a fresh load without cache:

```javascript
// Use page.route() which disables HTTP cache
await page.route('**/*', route => route.continue());
await page.goto('http://127.0.0.1:8000');
```

This is equivalent to a hard refresh (Cmd+Shift+R on Mac). Standard `page.reload()` uses cached resources.
