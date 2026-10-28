"""Exact and near-duplicate detection: MinHash LSH over character 5-grams.

LSH (datasketch) only proposes candidate pairs; each candidate is verified with
the exact Jaccard similarity of the two shingle sets, so reported pairs always
satisfy J >= threshold exactly. Texts are normalized with common.normalize first.
"""
from __future__ import annotations

from datasketch import MinHash, MinHashLSH

from common import JACCARD_THRESHOLD, LSH_CANDIDATE_THRESHOLD, NUM_PERM, SHINGLE_K


def shingles(norm_text: str, k: int = SHINGLE_K) -> frozenset[str]:
    if len(norm_text) < k:
        return frozenset([norm_text])
    return frozenset(norm_text[i : i + k] for i in range(len(norm_text) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class Index:
    """Shingle sets + MinHash signatures for a list of unique normalized texts."""

    def __init__(self, norm_texts: list[str], lsh_threshold: float = LSH_CANDIDATE_THRESHOLD):
        self.texts = norm_texts
        self.sets = [shingles(t) for t in norm_texts]
        # Sorted shingles make the byte sequence fed to MinHash independent of set order.
        self.minhashes = MinHash.bulk([[s.encode("utf-8") for s in sorted(ss)] for ss in self.sets], num_perm=NUM_PERM)
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=NUM_PERM)
        with self.lsh.insertion_session() as session:
            for i, m in enumerate(self.minhashes):
                session.insert(i, m)

    def query(self, minhash: MinHash) -> list[int]:
        return sorted(self.lsh.query(minhash))


def self_pairs(index: Index, threshold: float = JACCARD_THRESHOLD) -> list[tuple[int, int, float]]:
    """Verified pairs (i < j, J) within one index."""
    out = []
    for i, m in enumerate(index.minhashes):
        for j in index.query(m):
            if j <= i:
                continue
            jac = jaccard(index.sets[i], index.sets[j])
            if jac >= threshold:
                out.append((i, j, jac))
    return out


def cross_pairs(a: Index, b: Index, threshold: float = JACCARD_THRESHOLD) -> list[tuple[int, int, float]]:
    """Verified pairs (i in a, j in b, J)."""
    out = []
    for i, m in enumerate(a.minhashes):
        for j in b.query(m):
            jac = jaccard(a.sets[i], b.sets[j])
            if jac >= threshold:
                out.append((i, j, jac))
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Smaller index becomes the root: deterministic regardless of pair order.
            if rx < ry:
                self.parent[ry] = rx
            else:
                self.parent[rx] = ry
