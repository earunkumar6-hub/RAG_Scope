import pytest
from pydantic import ValidationError

from app.schemas.config import PipelineDefaults, RuntimeConfig
from app.schemas.errors import JEVError, jev_error_from
from app.schemas.params import IngestOverrides, QueryOverrides
from app.schemas.query import QueryRequest


def fields(exc: ValidationError, prefix: tuple = ()) -> dict[str, str]:
    return {d.field: d.message for d in jev_error_from(exc, prefix).details}


def test_valid_query_request_defaults():
    req = QueryRequest.model_validate({"query": "  What is GraphRAG?  "})
    assert req.query == "What is GraphRAG?"
    assert req.collection == "chunks"
    assert req.params == QueryOverrides()


@pytest.mark.parametrize("query", ["ab", "x" * 2001, "   a  "])
def test_query_length_bounds(query):
    with pytest.raises(ValidationError) as exc:
        QueryRequest.model_validate({"query": query})
    assert "query" in fields(exc.value)


def test_top_n_greater_than_top_k_reports_nested_field():
    with pytest.raises(ValidationError) as exc:
        QueryRequest.model_validate({"query": "abc", "params": {"top_k": 3, "top_n": 5}})
    assert fields(exc.value) == {"params.top_n": "top_n must be <= top_k (3)"}


@pytest.mark.parametrize(
    "body, field",
    [
        ({"query": "abc", "bogus": 1}, "bogus"),
        ({"query": "abc", "params": {"bogus": 1}}, "params.bogus"),
        ({"query": "abc", "filters": {"bogus": 1}}, "filters.bogus"),
    ],
)
def test_extra_keys_forbidden_at_every_level(body, field):
    with pytest.raises(ValidationError) as exc:
        QueryRequest.model_validate(body)
    assert field in fields(exc.value)


@pytest.mark.parametrize(
    "params, field",
    [
        ({"top_k": 0}, "params.top_k"),
        ({"top_k": 51}, "params.top_k"),
        ({"top_n": 21}, "params.top_n"),
        ({"similarity_threshold": 1.5}, "params.similarity_threshold"),
        ({"temperature": -0.1}, "params.temperature"),
        ({"seed": -1}, "params.seed"),
        ({"graph_hops": 4}, "params.graph_hops"),
        ({"hybrid_weight_vector": 2}, "params.hybrid_weight_vector"),
    ],
)
def test_param_ranges(params, field):
    with pytest.raises(ValidationError) as exc:
        QueryRequest.model_validate({"query": "abc", "params": params})
    assert field in fields(exc.value)


def test_invalid_session_id():
    with pytest.raises(ValidationError) as exc:
        QueryRequest.model_validate({"query": "abc", "session_id": "not-a-uuid"})
    assert "session_id" in fields(exc.value)


def test_resolve_query_merges_defaults():
    resolved = PipelineDefaults().resolve_query(QueryOverrides(top_k=20, temperature=0.0))
    assert resolved.top_k == 20 and resolved.temperature == 0.0
    assert resolved.top_n == 4 and resolved.seed == 42


def test_resolve_query_rechecks_cross_field_after_merge():
    # Client sends only top_k=2; default top_n=4 would violate top_n <= top_k.
    with pytest.raises(JEVError) as exc:
        PipelineDefaults().resolve_query(QueryOverrides(top_k=2))
    assert [d.field for d in exc.value.details] == ["params.top_n"]


def test_ingest_overlap_rule():
    with pytest.raises(ValidationError) as exc:
        IngestOverrides(chunk_size=256, chunk_overlap=129)
    assert fields(exc.value, ("params",)) == {
        "params.chunk_overlap": "chunk_overlap must be <= chunk_size / 2 (128)"
    }
    assert IngestOverrides(chunk_size=256, chunk_overlap=128).chunk_overlap == 128


def test_resolve_ingest_rechecks_overlap_against_default_size():
    # Default chunk_size=512 → max overlap 256.
    with pytest.raises(JEVError) as exc:
        PipelineDefaults().resolve_ingest(IngestOverrides(chunk_overlap=300))
    assert [d.field for d in exc.value.details] == ["params.chunk_overlap"]
    ok = PipelineDefaults().resolve_ingest(IngestOverrides(chunk_size=1024, chunk_overlap=300))
    assert (ok.chunk_size, ok.chunk_overlap, ok.build_graph) == (1024, 300, True)


def test_ingest_dedup_threshold_range():
    with pytest.raises(ValidationError):
        IngestOverrides(dedup_threshold=0.5)


def test_runtime_config_defaults_match_spec():
    d = RuntimeConfig().defaults
    assert (d.chunk_size, d.chunk_overlap, d.top_k, d.top_n) == (512, 64, 10, 4)
    assert (d.similarity_threshold, d.temperature, d.seed) == (0.35, 0.2, 42)
    assert (d.dedup_threshold, d.graph_hops, d.hybrid_weight_vector) == (0.95, 1, 0.7)
    assert RuntimeConfig().guardrails.input.prompt_injection.threshold == 0.8
    assert RuntimeConfig().guardrails.output.groundedness.max_unsupported_ratio == 0.3
