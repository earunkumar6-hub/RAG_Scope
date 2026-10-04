"""Select the GraphStore backend: Neo4j when ``NEO4J_URI`` is set, otherwise local NetworkX."""

from app.core.config import Settings
from app.graph.base import GraphStore


def build_graph_store(settings: Settings) -> GraphStore:
    if settings.graph_backend == "neo4j":
        from app.graph.neo4j_store import Neo4jGraphStore

        password = settings.neo4j_password.get_secret_value() if settings.neo4j_password else ""
        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, password)
    from app.graph.networkx_store import NetworkXGraphStore

    return NetworkXGraphStore(settings.data_dir)
