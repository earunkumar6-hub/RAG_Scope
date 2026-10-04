"""S6 similar-data segregation: near-duplicate detection and topic clustering.

* Duplicates: a new chunk whose cosine similarity to an existing chunk (or an earlier chunk in
  the same batch) is >= ``dedup_threshold`` is linked to the *root* canonical chunk. Duplicates are
  still stored (flagged) so provenance survives; retrieval collapses them and KG extraction skips
  them.
* Clusters: seeded k-means over all canonical chunks of the corpus, k chosen by silhouette score.
  Inputs are sorted by chunk id and clusters renumbered by their smallest member id, so the same
  corpus + seed always yields the same ids. Ids may change when the corpus changes.
* Labels: 3-word LLM label per cluster (cached by membership); TF-IDF keywords when no LLM.
"""

import hashlib
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.llm.base import LLMError, LLMProvider
from app.pipeline.ingestion.stop_words import ENGLISH_STOP_WORDS
from app.pipeline.ingestion.vector_store import Neighbour

logger = logging.getLogger(__name__)

MAX_CLUSTERS = 12
SILHOUETTE_SAMPLE = 2000
TOKEN_RX = re.compile(r"\b[a-z][a-z]{2,}\b")  # applied to lower-cased text
LABEL_SAMPLES = 5
LABEL_SAMPLE_CHARS = 400


@dataclass(frozen=True)
class DuplicateMatch:
    chunk_id: str
    duplicate_of: str  # root canonical chunk id
    similarity: float
    source: str  # "existing" | "batch"


def find_duplicates(
    ids: list[str],
    vectors: np.ndarray,
    existing: list[Neighbour | None],
    threshold: float,
) -> dict[str, DuplicateMatch]:
    """Map new chunk id -> match, for chunks that are near-duplicates."""
    matches: dict[str, DuplicateMatch] = {}
    sims = vectors @ vectors.T if len(ids) else np.zeros((0, 0))
    for i, chunk_id in enumerate(ids):
        best: DuplicateMatch | None = None
        nb = existing[i]
        if nb is not None and nb.similarity >= threshold:
            root = nb.metadata.get("is_duplicate_of") or nb.chunk_id
            best = DuplicateMatch(chunk_id, root, nb.similarity, "existing")
        if i > 0:
            j = int(np.argmax(sims[i, :i]))
            sim = float(sims[i, j])
            if sim >= threshold and (best is None or sim > best.similarity):
                prior = matches.get(ids[j])
                root = prior.duplicate_of if prior else ids[j]
                best = DuplicateMatch(chunk_id, root, sim, "batch")
        if best is not None:
            matches[chunk_id] = best
    return matches


def _sq_dists(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = (x * x).sum(1)[:, None] - 2.0 * x @ centers.T + (centers * centers).sum(1)[None, :]
    return np.maximum(d, 0.0)


def kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 10, max_iter: int = 100) -> np.ndarray:
    """Seeded k-means++ / Lloyd in numpy; best of ``n_init`` runs by inertia.

    Implemented here instead of ``sklearn.cluster``: deterministic for a given seed, and it avoids
    importing sklearn's compiled linear-model extensions, which some Windows application-control
    policies block.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = np.zeros(len(x), dtype=int), np.inf
    for _ in range(n_init):
        centers = x[[rng.integers(len(x))]]
        for _ in range(1, k):  # k-means++ seeding
            d2 = _sq_dists(x, centers).min(axis=1)
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(len(x), 1.0 / len(x))
            centers = np.vstack([centers, x[rng.choice(len(x), p=probs)]])
        labels = _sq_dists(x, centers).argmin(axis=1)
        for _ in range(max_iter):
            for j in range(k):
                members = x[labels == j]
                if len(members):
                    centers[j] = members.mean(axis=0)
            new_labels = _sq_dists(x, centers).argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        inertia = float(_sq_dists(x, centers)[np.arange(len(x)), labels].sum())
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def silhouette_cosine(x: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    """Mean silhouette coefficient with cosine distance, in numpy.

    Matches sklearn's ``silhouette_score(metric="cosine", sample_size=..., random_state=seed)``
    (same seeded sample), without importing sklearn (see ``kmeans``).
    """
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    if len(x) > sample_size:
        idx = np.random.RandomState(seed).permutation(len(x))[:sample_size]
        x, labels = x[idx], labels[idx]
    unit = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    dist = np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    clusters = np.unique(labels)
    member = labels[:, None] == clusters[None, :]  # (n, k)
    sizes = member.sum(0)
    sums = dist @ member  # (n, k): summed distance from each point to each cluster
    own = member.argmax(1)
    own_size = sizes[own]
    a = sums[np.arange(len(x)), own] / np.maximum(own_size - 1, 1)
    other = np.where(member, np.inf, sums / sizes[None, :])
    b = other.min(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.nan_to_num((b - a) / np.maximum(a, b))
    s[own_size == 1] = 0.0  # singleton clusters score 0
    return float(s.mean())


def cluster_vectors(ids: list[str], vectors: np.ndarray, seed: int) -> dict[str, int]:
    """Deterministic k-means clustering with k chosen by silhouette; returns id -> cluster id."""
    n = len(ids)
    if n == 0:
        return {}
    order = sorted(range(n), key=lambda i: ids[i])
    sorted_ids = [ids[i] for i in order]
    x = np.asarray(vectors)[order]
    if n < 3:
        return dict.fromkeys(sorted_ids, 0)

    best_labels: np.ndarray | None = None
    best_score = -2.0
    for k in range(2, min(MAX_CLUSTERS, n - 1) + 1):
        labels = kmeans(x, k, seed)
        if len(set(labels)) < 2:
            continue
        score = silhouette_cosine(x, labels, SILHOUETTE_SAMPLE, seed)
        if score > best_score:
            best_score, best_labels = float(score), labels
    if best_labels is None:
        return dict.fromkeys(sorted_ids, 0)

    renumber: dict[int, int] = {}
    for label in best_labels:  # ids are sorted, so first occurrence = smallest member id
        renumber.setdefault(int(label), len(renumber))
    return {cid: renumber[int(lbl)] for cid, lbl in zip(sorted_ids, best_labels, strict=True)}


def member_hash(chunk_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(chunk_ids)).encode()).hexdigest()[:16]


def keyword_labels(texts_by_cluster: dict[int, list[str]], words: int = 3) -> dict[int, str]:
    """Top TF-IDF terms per cluster (class-based: each cluster's text is one document).

    Same weighting as sklearn's ``TfidfVectorizer(stop_words="english", sublinear_tf=True)``:
    tf = 1 + ln(count), idf = ln((1 + n) / (1 + df)) + 1, rows L2-normalised.
    """
    cluster_ids = sorted(texts_by_cluster)
    counts = [
        Counter(
            t
            for t in TOKEN_RX.findall(" ".join(texts_by_cluster[c]).lower())
            if t not in ENGLISH_STOP_WORDS
        )
        for c in cluster_ids
    ]
    terms = sorted(set().union(*counts))
    if not terms:  # empty vocabulary
        return {c: f"Topic {c}" for c in cluster_ids}
    col = {t: i for i, t in enumerate(terms)}
    tf = np.zeros((len(cluster_ids), len(terms)))
    for row, cnt in enumerate(counts):
        for t, n in cnt.items():
            tf[row, col[t]] = 1.0 + np.log(n)
    df = (tf > 0).sum(0)
    matrix = tf * (np.log((1 + len(cluster_ids)) / (1 + df)) + 1.0)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    labels: dict[int, str] = {}
    for row, cid in enumerate(cluster_ids):
        scores = matrix[row]
        top = [terms[i] for i in np.argsort(-scores, kind="stable")[:words] if scores[i] > 0]
        labels[cid] = " ".join(t.capitalize() for t in top) or f"Topic {cid}"
    return labels


LABEL_SYSTEM = (
    "You name topic clusters of document excerpts. Reply with a JSON object "
    '{"label": "<exactly three words, Title Case>"} and nothing else.'
)


def llm_label(llm: LLMProvider, samples: list[str], seed: int) -> str:
    """Ask the LLM for a 3-word label; raises ``LLMError`` on bad output."""
    excerpts = "\n\n".join(f"- {s[:LABEL_SAMPLE_CHARS]}" for s in samples[:LABEL_SAMPLES])
    data = llm.complete_json(
        [
            {"role": "system", "content": LABEL_SYSTEM},
            {"role": "user", "content": f"Excerpts from one cluster:\n\n{excerpts}"},
        ],
        temperature=0.0,
        seed=seed,
        max_tokens=30,
        fast=True,
    )
    label = data.get("label")
    if not isinstance(label, str) or not 1 <= len(label.split()) <= 4:
        raise LLMError(f"invalid label {label!r}")
    return label.strip()


def label_clusters(
    texts_by_cluster: dict[int, list[str]],
    hashes: dict[int, str],
    cached: dict[str, tuple[str, str]],
    llm_factory: Callable[[], LLMProvider],
    seed: int,
) -> tuple[dict[int, tuple[str, str]], str | None]:
    """Label each cluster -> (label, source). Reuses cached labels for unchanged membership.

    Returns the labels and, if the LLM could not be used, the reason (keyword fallback applied).
    """
    labels: dict[int, tuple[str, str]] = {}
    todo = []
    for cid in sorted(texts_by_cluster):
        hit = cached.get(hashes[cid])
        if hit is not None:
            labels[cid] = hit
        else:
            todo.append(cid)
    if not todo:
        return labels, None
    keywords = keyword_labels(texts_by_cluster)
    fallback_reason: str | None = None
    try:
        llm = llm_factory()
    except LLMError as exc:
        llm, fallback_reason = None, str(exc)
    for cid in todo:
        if llm is not None:
            try:
                labels[cid] = (llm_label(llm, texts_by_cluster[cid], seed), "llm")
                continue
            # Labels are cosmetic: any provider failure (including a malformed response the
            # SDK cannot parse) falls back to keywords instead of failing the whole ingest.
            except Exception as exc:
                fallback_reason = (
                    str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
                )
                logger.warning("cluster label fallback", extra={"cluster": cid, "error": str(exc)})
        labels[cid] = (keywords[cid], "keywords")
    return labels, fallback_reason
