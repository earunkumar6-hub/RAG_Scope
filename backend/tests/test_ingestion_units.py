import httpx
import numpy as np
import pytest

from app.llm.base import LLMNotConfigured
from app.llm.openai_provider import OpenAIProvider
from app.pipeline.ingestion.chunker import chunk_text
from app.pipeline.ingestion.embedder import l2_normalise, pca_2d
from app.pipeline.ingestion.parser import parse_document, remove_repeated_headers_footers
from app.pipeline.ingestion.segregator import (
    cluster_vectors,
    find_duplicates,
    keyword_labels,
    label_clusters,
    member_hash,
    silhouette_cosine,
)
from app.pipeline.ingestion.tokenizer import get_tokenizer
from app.pipeline.ingestion.upload import sniff_error
from app.pipeline.ingestion.vector_store import EmbeddingMismatchError, Neighbour, VectorStore
from tests.helpers import FakeEmbedder, make_docx, make_pdf, make_text

TOK = get_tokenizer("cl100k_base")


# ---------------------------------------------------------------- upload sniffing
def test_sniff_accepts_real_files_and_rejects_mismatches():
    assert sniff_error(".pdf", make_pdf(["hello"])) is None
    assert sniff_error(".docx", make_docx(["hello"])) is None
    assert sniff_error(".txt", "héllo".encode()) is None
    assert "PDF" in sniff_error(".pdf", b"just text")
    assert "ZIP" in sniff_error(".docx", b"%PDF-1.4")
    assert "binary" in sniff_error(".txt", b"abc\x00def")
    assert "UTF-8" in sniff_error(".md", b"\xff\xfe\xfa")
    assert sniff_error(".txt", b"") == "file is empty"


# ---------------------------------------------------------------- parsing
def test_pdf_header_footer_removed_and_page_map():
    pages = [make_text(seed=i, paragraphs=2) for i in range(4)]
    parsed = parse_document(".pdf", make_pdf(pages, header="ACME Confidential Report"))
    assert parsed.page_count == 4
    assert "ACME Confidential Report" not in parsed.text
    assert "Page 2" not in parsed.text
    assert any("ACME" in line for line in parsed.removed_lines)
    assert [p.page for p in parsed.pages] == [1, 2, 3, 4]
    assert parsed.page_at(parsed.pages[2].start + 1) == 3
    assert parsed.raw_chars > len(parsed.text)


def test_header_rule_needs_majority_and_three_pages():
    pages = ["Title A\nbody one", "Title A\nbody two", "Other\nbody three", "Other2\nbody four"]
    cleaned, removed = remove_repeated_headers_footers(pages)
    assert removed == []  # "Title A" on 2/4 pages is not > 50%
    assert remove_repeated_headers_footers(pages[:2]) == (pages[:2], [])


def test_markdown_docx_and_formfeed_txt():
    md = parse_document(".md", b"# Heading\n\nSome **bold** text.\n\n- item one\n- item two")
    assert "**" not in md.text and "#" not in md.text
    assert "Heading" in md.text and "item two" in md.text

    dx = parse_document(".docx", make_docx(["First para.", "Second para."], [["k", "v"]]))
    assert "First para." in dx.text and "k | v" in dx.text

    txt = parse_document(".txt", b"page one text\fpage two text")
    assert txt.page_count == 2 and txt.page_at(len(txt.text) - 1) == 2


def test_normalise_whitespace_and_hyphenation():
    parsed = parse_document(".txt", b"multi-\nline   spaced\r\n\r\n\r\n\r\nnext\tpara")
    assert parsed.text == "multiline spaced\n\nnext para"


# ---------------------------------------------------------------- chunking
@pytest.mark.parametrize("size,overlap", [(128, 0), (128, 16), (256, 64), (512, 64), (200, 100)])
def test_chunker_invariants(size, overlap):
    text = make_text(seed=size, paragraphs=40)
    tokens = TOK.encode(text)
    spans = chunk_text(text, TOK, size, overlap)
    assert spans[0].start_token == 0 and spans[-1].end_token == len(tokens)
    for prev, cur in zip(spans, spans[1:], strict=False):
        assert cur.start_token == prev.end_token - overlap  # exact token overlap
        assert cur.overlap_prev_tokens == overlap
        assert cur.overlap_prev_chars == prev.end_char - cur.start_char >= 0
    for sp in spans:
        assert 0 < sp.token_count <= size
        assert sp.text == text[sp.start_char : sp.end_char]
    for sp in spans[:-1]:
        assert sp.token_count >= max(size // 2, overlap + 1)


def test_chunker_prefers_paragraph_boundaries_and_is_deterministic():
    text = make_text(seed=3, paragraphs=30)
    spans = chunk_text(text, TOK, 256, 0)
    # cl100k merges ".\n\n" into one token, so a paragraph split leaves the break at the chunk end
    ends_on_paragraph = sum(text[: sp.end_char].endswith("\n\n") for sp in spans[:-1])
    assert ends_on_paragraph >= len(spans[:-1]) // 2
    assert spans == chunk_text(text, TOK, 256, 0)


def test_chunk_size_changes_count():
    text = make_text(seed=5, paragraphs=40)
    assert len(chunk_text(text, TOK, 128, 16)) > len(chunk_text(text, TOK, 512, 16))


def test_chunker_empty_text():
    assert chunk_text("", TOK, 128, 0) == []


# ---------------------------------------------------------------- embedding helpers
def test_l2_normalise_and_pca_are_deterministic():
    rng = np.random.default_rng(0)
    v = l2_normalise(rng.normal(size=(20, 8)))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(pca_2d(v), pca_2d(v.copy()))
    assert pca_2d(v[:1]).shape == (1, 2)


# ---------------------------------------------------------------- segregation
def test_find_duplicates_against_existing_and_within_batch():
    e = FakeEmbedder()
    texts = ["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma", "eta theta iota"]
    vecs = e.embed_documents(texts)
    existing = [
        None,
        None,
        None,
        Neighbour("old-1", 1.0, {"is_duplicate_of": "old-0"}),  # chain -> root old-0
    ]
    dups = find_duplicates(["n0", "n1", "n2", "n3"], vecs, existing, threshold=0.95)
    assert set(dups) == {"n2", "n3"}
    assert dups["n2"].duplicate_of == "n0" and dups["n2"].source == "batch"
    assert dups["n3"].duplicate_of == "old-0" and dups["n3"].source == "existing"


def test_cluster_vectors_deterministic_and_renumbered():
    e = FakeEmbedder()
    texts = [
        make_text(seed=i, paragraphs=1, topic_cycle=[t])
        for i, t in enumerate(["graphs", "cooking"] * 6)
    ]
    ids = [f"c{i:02d}" for i in range(len(texts))]
    vecs = e.embed_documents(texts)
    a = cluster_vectors(ids, vecs, seed=42)
    b = cluster_vectors(list(reversed(ids)), vecs[::-1], seed=42)
    assert a == b
    assert a["c00"] == 0  # smallest id -> cluster 0
    assert a["c00"] == a["c02"] and a["c00"] != a["c01"]
    assert cluster_vectors(ids[:2], vecs[:2], seed=1) == {"c00": 0, "c01": 0}


def test_silhouette_cosine_matches_definition():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 5))
    labels = np.array([0] * 12 + [1] * 17 + [2])  # cluster 2 is a singleton

    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = 1 - unit @ unit.T
    expected = []
    for i in range(len(x)):
        same = (labels == labels[i]) & (np.arange(len(x)) != i)
        if not same.any():
            expected.append(0.0)
            continue
        a = d[i, same].mean()
        b = min(d[i, labels == c].mean() for c in set(labels) - {labels[i]})
        expected.append((b - a) / max(a, b))
    assert silhouette_cosine(x, labels, 2000, 42) == pytest.approx(np.mean(expected))
    # sampling is seeded: same seed, same score
    assert silhouette_cosine(x, labels, 20, 7) == silhouette_cosine(x, labels, 20, 7)


def test_keyword_labels():
    labels = keyword_labels({0: ["flour butter oven flour"], 1: ["ledger audit ledger invoice"]})
    assert "Flour" in labels[0] and "Ledger" in labels[1]


def _mock_openai(content: str) -> OpenAIProvider:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    return OpenAIProvider(
        "sk-test", "m", "m-fast", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_label_clusters_llm_cache_and_fallbacks():
    texts = {0: ["flour butter oven"], 1: ["ledger audit invoice"]}
    hashes = {0: member_hash(["a"]), 1: member_hash(["b"])}

    labels, reason = label_clusters(
        texts, hashes, {}, lambda: _mock_openai('{"label": "Home Baking Basics"}'), 42
    )
    assert labels[0] == ("Home Baking Basics", "llm") and reason is None

    labels, reason = label_clusters(texts, hashes, {}, lambda: _mock_openai("not json at all"), 42)
    assert labels[0][1] == "keywords" and "invalid JSON" in reason

    def no_llm():
        raise LLMNotConfigured("OPENAI_API_KEY not set")

    cached = {hashes[0]: ("Cached Label Here", "llm")}
    labels, reason = label_clusters(texts, hashes, cached, no_llm, 42)
    assert labels[0] == ("Cached Label Here", "llm")
    assert labels[1][1] == "keywords" and reason == "OPENAI_API_KEY not set"


# ---------------------------------------------------------------- vector store
def test_vector_store_cosine_and_embedding_mismatch(tmp_path):
    vs = VectorStore(tmp_path / "chroma", "chunks")
    col = vs.collection("fake", 3)
    vs.upsert(
        col,
        ["a", "b"],
        np.array([[1.0, 0, 0], [0, 1.0, 0]]),
        ["A", "B"],
        [
            {"document_id": "d1", "is_duplicate_of": ""},
            {"document_id": "d2", "is_duplicate_of": ""},
        ],
    )
    [nb] = vs.nearest(col, np.array([[0, 1.0, 0]]), [])
    assert nb.chunk_id == "b" and nb.similarity == pytest.approx(1.0)
    [nb] = vs.nearest(col, np.array([[0, 1.0, 0]]), ["d2"])
    assert nb.chunk_id == "a" and nb.similarity == pytest.approx(0.0, abs=1e-6)
    ids, vecs, texts = vs.unique_records(col, ["d1"])
    assert ids == ["b"] and vecs.shape == (1, 3) and texts == ["B"]
    with pytest.raises(EmbeddingMismatchError):
        vs.collection("other-model", 3)


def test_label_clusters_falls_back_on_unexpected_provider_error():
    class Broken:
        def complete_json(self, *args, **kwargs):
            raise AttributeError("'str' object has no attribute 'usage'")

    texts = {0: ["flour butter oven"]}
    labels, reason = label_clusters(texts, {0: member_hash(["a"])}, {}, lambda: Broken(), 42)
    assert labels[0][1] == "keywords"
    assert reason == "AttributeError: 'str' object has no attribute 'usage'"
