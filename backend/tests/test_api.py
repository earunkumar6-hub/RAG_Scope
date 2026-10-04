from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.runtime_config import RuntimeConfigStore
from app.main import create_app
from app.schemas.query import QueryRequest


def test_health_reports_every_component_without_failing(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["components"]) == {
        "database",
        "chroma",
        "graph_store",
        "llm",
        "embedding_model",
    }
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["llm"]["status"] == "not_configured"  # no key in tests
    assert body["status"] == "degraded"


def test_health_llm_configured_with_key(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, openai_api_key="sk-test")
    with TestClient(create_app(settings)) as c:
        llm = c.get("/api/health").json()["components"]["llm"]
    assert llm["status"] == "configured"
    assert "sk-test" not in str(llm)


def test_env_example_loads_with_empty_values_as_unset(tmp_path):
    from app.core.config import REPO_ROOT

    settings = Settings(_env_file=REPO_ROOT / ".env.example", data_dir=tmp_path)
    assert settings.openai_api_key is None and settings.neo4j_uri is None
    assert settings.graph_backend == "networkx"
    assert settings.cors_origin_list == ["http://localhost:3000"]
    assert settings.openai_model == "gpt-4.1-mini"


def test_schema_whitelist(client: TestClient):
    names = client.get("/api/schema").json()
    assert "QueryRequest" in names
    schema = client.get("/api/schema/QueryRequest").json()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["query"]

    resp = client.get("/api/schema/Settings")
    assert resp.status_code == 404
    assert resp.json()["error"] == "NOT_FOUND"


def test_config_roundtrip_and_persistence(client: TestClient, settings: Settings):
    cfg = client.get("/api/config").json()
    assert cfg["defaults"]["top_k"] == 10
    assert "api_key" not in str(cfg).lower()

    cfg["defaults"]["top_k"] = 20
    cfg["guardrails"]["input"]["toxicity"]["enabled"] = False
    resp = client.put("/api/config", json=cfg)
    assert resp.status_code == 200
    assert client.get("/api/config").json()["defaults"]["top_k"] == 20

    reloaded = RuntimeConfigStore(settings.runtime_config_path).get()
    assert reloaded.defaults.top_k == 20
    assert reloaded.guardrails.input.toxicity.enabled is False


def test_config_put_validation_envelope(client: TestClient):
    cfg = client.get("/api/config").json()
    cfg["defaults"]["top_k"] = 2  # default top_n is 4
    cfg["defaults"]["chunk_overlap"] = 400  # > 512 / 2
    resp = client.put("/api/config", json=cfg)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert {d["field"] for d in body["details"]} == {"defaults.top_n", "defaults.chunk_overlap"}


def test_config_put_partial_defaults_still_cross_validated(client: TestClient):
    resp = client.put("/api/config", json={"defaults": {"top_k": 2}})
    assert resp.status_code == 422
    assert [d["field"] for d in resp.json()["details"]] == ["defaults.top_n"]


def _app_with_probe(app: FastAPI) -> FastAPI:
    """Attach a throwaway route that parses a QueryRequest and resolves it against defaults."""

    @app.post("/_probe")
    def probe(req: QueryRequest) -> dict:
        resolved = app.state.runtime_config.get().defaults.resolve_query(req.params)
        return resolved.model_dump()

    return app


def test_request_body_error_envelope_strips_body_prefix(app: FastAPI):
    with TestClient(_app_with_probe(app)) as c:
        resp = c.post("/_probe", json={"query": "abc", "params": {"top_k": 3, "top_n": 5}})
    assert resp.status_code == 422
    assert resp.json() == {
        "error": "VALIDATION_ERROR",
        "details": [{"field": "params.top_n", "message": "top_n must be <= top_k (3)"}],
    }


def test_merged_defaults_error_uses_same_envelope(app: FastAPI):
    with TestClient(_app_with_probe(app)) as c:
        resp = c.post("/_probe", json={"query": "abc", "params": {"top_k": 2}})
        ok = c.post("/_probe", json={"query": "abc", "params": {"top_k": 2, "top_n": 2}})
    assert resp.status_code == 422
    assert resp.json()["details"][0]["field"] == "params.top_n"
    assert ok.status_code == 200 and ok.json()["top_n"] == 2


def test_malformed_json_and_missing_body(app: FastAPI):
    with TestClient(_app_with_probe(app)) as c:
        bad = c.post("/_probe", content=b"{not json", headers={"Content-Type": "application/json"})
        missing = c.post("/_probe")
    assert bad.status_code == 422 and bad.json()["error"] == "VALIDATION_ERROR"
    assert bad.json()["details"][0]["field"] == "body"
    assert missing.status_code == 422 and missing.json()["details"][0]["field"] == "body"


def test_outdated_runtime_config_file_does_not_block_startup(settings: Settings):
    settings.runtime_config_path.write_text('{"defaults": {"top_k": 7}, "removed_field": 1}')
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/config").json()["defaults"]["top_k"] == 10
    assert settings.runtime_config_path.with_suffix(".invalid.json").exists()
