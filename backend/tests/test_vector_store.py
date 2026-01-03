from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Tests for the SearchResults dataclass"""

    def test_from_chroma_extracts_documents(self):
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course_title": "Test"}]],
            "distances": [[0.1, 0.2]],
        }
        results = SearchResults.from_chroma(chroma_results)
        assert results.documents == ["doc1", "doc2"]

    def test_empty_creates_error_result(self):
        results = SearchResults.empty("Test error")
        assert results.is_empty()
        assert results.error == "Test error"

    def test_is_empty_returns_true_for_no_documents(self):
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty()

    def test_is_empty_returns_false_for_documents(self):
        results = SearchResults(documents=["content"], metadata=[{}], distances=[0.1])
        assert not results.is_empty()


class TestVectorStoreMaxResults:
    """Tests to verify MAX_RESULTS=0 causes the failure"""

    def test_search_with_max_results_zero_returns_error(self, tmp_path):
        """Test that max_results=0 returns an error (ChromaDB rejects 0)"""
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,  # Invalid value - ChromaDB rejects 0
        )

        # Add test data with all required fields (ChromaDB doesn't accept None)
        from models import Course, CourseChunk, Lesson

        course = Course(
            title="Test Course",
            course_link="http://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Intro",
                    lesson_link="http://example.com/lesson1",
                )
            ],
        )
        store.add_course_metadata(course)
        store.add_course_content(
            [
                CourseChunk(
                    content="AI content about machine learning",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]
        )

        # max_results=0 causes ChromaDB to return an error
        results = store.search("AI")
        assert results.is_empty(), "Search with max_results=0 should fail"
        assert results.error is not None, "Error should be set"
        assert "cannot be negative, or zero" in results.error

    def test_search_with_valid_max_results(self, tmp_path):
        """This test should PASS with MAX_RESULTS=5"""
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,  # Valid value
        )

        from models import Course, CourseChunk, Lesson

        course = Course(
            title="Test Course",
            course_link="http://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Intro",
                    lesson_link="http://example.com/lesson1",
                )
            ],
        )
        store.add_course_metadata(course)
        store.add_course_content(
            [
                CourseChunk(
                    content="AI content about machine learning",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]
        )

        results = store.search("AI")
        assert not results.is_empty(), "Search should return results"


class TestVectorStoreSearch:
    """Tests for VectorStore search functionality"""

    def test_search_with_course_filter(self, tmp_path):
        """Test filtering by course name"""
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

        from models import Course, CourseChunk, Lesson

        course = Course(
            title="Python Basics",
            course_link="http://example.com/python",
            instructor="Python Teacher",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Intro",
                    lesson_link="http://example.com/lesson1",
                )
            ],
        )
        store.add_course_metadata(course)
        store.add_course_content(
            [
                CourseChunk(
                    content="Python programming fundamentals",
                    course_title="Python Basics",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]
        )

        results = store.search("Python", course_name="Python Basics")
        assert not results.is_empty()

    def test_search_invalid_course_returns_error(self, tmp_path):
        """Test that invalid course name returns error"""
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

        results = store.search("test", course_name="Nonexistent Course")
        assert "No course found" in results.error


class TestVectorStoreFilterBuilding:
    """Tests for the _build_filter method"""

    def test_build_filter_no_params(self, tmp_path):
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        result = store._build_filter(None, None)
        assert result is None

    def test_build_filter_course_only(self, tmp_path):
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        result = store._build_filter("Test Course", None)
        assert result == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, tmp_path):
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        result = store._build_filter(None, 1)
        assert result == {"lesson_number": 1}

    def test_build_filter_both_params(self, tmp_path):
        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        result = store._build_filter("Test Course", 1)
        assert result == {
            "$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]
        }
