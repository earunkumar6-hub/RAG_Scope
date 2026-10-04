"""Q6 hybrid fusion: weighted Reciprocal Rank Fusion of the vector and graph candidate lists.

score(d) = w / (k + rank_vector(d)) + (1 - w) / (k + rank_graph(d)), ranks 1-based, a missing
rank contributes 0; ``w`` = hybrid_weight_vector, k = 60 (the usual RRF constant).
"""

from dataclasses import dataclass

RRF_K = 60


@dataclass(frozen=True)
class FusedRow:
    chunk_id: str
    vector_rank: int | None
    graph_rank: int | None
    score: float

    @property
    def source(self) -> str:
        if self.vector_rank and self.graph_rank:
            return "both"
        return "vector" if self.vector_rank else "graph"


def weighted_rrf(
    vector_ids: list[str], graph_ids: list[str], weight_vector: float, k: int = RRF_K
) -> list[FusedRow]:
    """Fused rows best first, de-duplicated by chunk id. Ties: vector rank, then graph rank.

    Rows scoring 0 are dropped, so weight 1.0 means vector-only and 0.0 graph-only.
    """
    v_rank = {cid: i for i, cid in enumerate(dict.fromkeys(vector_ids), start=1)}
    g_rank = {cid: i for i, cid in enumerate(dict.fromkeys(graph_ids), start=1)}
    rows = []
    for cid in dict.fromkeys([*v_rank, *g_rank]):
        vr, gr = v_rank.get(cid), g_rank.get(cid)
        score = (weight_vector / (k + vr) if vr else 0.0) + (
            (1 - weight_vector) / (k + gr) if gr else 0.0
        )
        if score > 0:
            rows.append(FusedRow(cid, vr, gr, score))
    big = len(rows) + 1
    rows.sort(key=lambda r: (-r.score, r.vector_rank or big, r.graph_rank or big))
    return rows
