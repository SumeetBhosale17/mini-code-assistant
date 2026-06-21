import numpy as np

from src.embedder import embed_texts


def test_shape_and_dtype():
    vecs = embed_texts(["hello world", "goodbye world"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32


def test_vectors_are_normalized():
    vec = embed_texts(["any text"])[0]
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)  # invariant # 2


def test_semantic_similarity():
    a, b, c = embed_texts(
        [
            "connect to a database",
            "establish a database connection",
            "bake a chocolate cake",
        ]
    )
    assert a @ b > a @ c  # normalized -> dot product is cosine; related > unrelated
