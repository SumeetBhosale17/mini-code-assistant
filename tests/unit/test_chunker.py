import textwrap

from src.chunker import chunk_repo, chunk_source


def test_extracts_each_top_level_def():
    """Each top-level function is extracted as its own chunk."""
    source = textwrap.dedent("""\
        def alpha():
            return 1

        def beta():
            return 2
    """)

    chunks = chunk_source(source, "x.py")

    # Names are extracted correctly
    assert [c.name for c in chunks] == ["alpha", "beta"]
    # Both are identified as functions
    assert all(c.chunk_type == "function" for c in chunks)


def test_class_is_one_chunk_including_methods():
    """A class is a single chunk that contains its methods."""
    source = textwrap.dedent("""\
        class Foo:
            def method_a(self): ...
            def method_b(self): ...
    """)
    chunks = chunk_source(source, "x.py")

    # Exactly one chunk for the whole class
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "class"

    # Methods live inside the class content, not as separate chunks
    assert "method_a" in chunks[0].content
    assert "method_b" in chunks[0].content


def test_decorator_is_included_in_chunk():
    """Decorators are included, and start_line points at the decorator."""
    source = textwrap.dedent("""\
        @app.route("/login")
        def login(): ...
    """)
    chunks = chunk_source(source, "x.py")

    assert len(chunks) == 1
    # Content includes the decorator
    assert chunks[0].content.lstrip().startswith("@app.route")
    # start_line is the decorator line, not the def line
    assert chunks[0].start_line == 1


def test_syntax_error_returns_empty():
    """Invalid Python returns [] instead of crashing."""
    # Missing parameter name causes a SyntaxError
    bad_source = "def broken(:\n"

    result = chunk_source(bad_source, "x.py")
    assert result == []


def test_chunk_repo_skips_pycache(tmp_path):
    """chunk_repo skips files in excluded directories like __pycache__."""
    # A valid Python file at the top level
    (tmp_path / "a.py").write_text("def f(): ...")

    # A __pycache__ directory with a Python file inside
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("def g(): ...")

    chunks = chunk_repo(str(tmp_path))
    names = [c.name for c in chunks]

    # Only 'f' is found; 'g' is skipped because it's in __pycache__
    assert names == ["f"]
    assert len(chunks) == 1
    assert chunks[0].file_path.endswith("a.py")
