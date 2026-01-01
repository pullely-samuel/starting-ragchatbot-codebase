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
