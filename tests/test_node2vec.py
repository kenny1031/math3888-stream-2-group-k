"""
Unit tests for weighted Node2Vec.
Run from repository root:
    python -m pytest tests/test_node2vec.py -v
"""
from collections import Counter
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import src.node2vec as node2vec
from src.node2vec import Node2VecTrainer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_graph():
    """
    A triangle A-B-C, with an outward edge B-D.
    When walking A -> B:
        A = return
        C = neighbour of A
        D = outward exploration
    """
    G = nx.Graph()

    for u, v in [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("B", "D"),
    ]:
        G.add_edge(u, v, confidence=1.0)

    return G


# ============================================================
# 1. Walk validity
# ============================================================

def test_walk_validity(small_graph):
    small_graph.add_node("Z")  # Isolated node
    trainer = Node2VecTrainer(
        walk_length=12,
        walks_per_node=4,
        seed=3888,
    )

    walks = trainer.generate_walks(small_graph)

    assert len(walks) == 4 * small_graph.number_of_nodes()
    assert all(1 <= len(walk) <= 12 for walk in walks)
    assert all(
        small_graph.has_edge(u, v)
        for walk in walks
        for u, v in zip(walk[:-1], walk[1:])
    )

    # An isolated node should generate a one-node walk.
    assert all(
        len(walk) == 1
        for walk in walks
        if walk[0] == "Z"
    )


# ============================================================
# 2. Weighted first-order sampling
# ============================================================

def test_weighted_sampling():

    G = nx.Graph()
    G.add_edge("A", "B", confidence=0.9)
    G.add_edge("A", "C", confidence=0.1)

    trainer = Node2VecTrainer(seed=42)
    trainer._prepare_graph(G)

    rng = np.random.default_rng(42)

    samples = [
        trainer._sample_next(None, "A", rng)
        for _ in range(12000)
    ]

    counts = Counter(samples)

    # Expected probabilities: B = 0.9, C = 0.1
    assert counts["B"] / len(samples) == pytest.approx(0.9, abs=0.02)
    assert counts["C"] / len(samples) == pytest.approx(0.1, abs=0.02)


# ============================================================
# 3. Second-order transition probabilities
# ============================================================

@pytest.mark.parametrize(
    "p,q,expected",
    [
        (2.0, 4.0, {"A": 2/7, "C": 4/7, "D": 1/7}),
        (1.0, 0.25, {"A": 1/6, "C": 1/6, "D": 2/3})
    ]
)
def test_second_order_bias(small_graph, p, q, expected):
    trainer = Node2VecTrainer(p=p, q=q, seed=42,)
    trainer._prepare_graph(small_graph)
    rng = np.random.default_rng(42)
    samples = [
        trainer._sample_next("A", "B", rng)
        for _ in range(20000)
    ]

    counts = Counter(samples)

    for node, probability in expected.items():
        observed = counts[node] / len(samples)
        assert observed == pytest.approx(probability, abs=0.025)


# ============================================================
# 4. Reproducibility of random walks
# ============================================================

def test_walk_reproducibility(small_graph):
    params = dict(
        walk_length=15,
        walks_per_node=5,
        p=0.5,
        q=2.0,
        seed=3888,
    )

    trainer1 = Node2VecTrainer(**params)
    trainer2 = Node2VecTrainer(**params)

    walks1 = trainer1.generate_walks(small_graph)
    walks2 = trainer2.generate_walks(small_graph)

    assert walks1 == walks2


# ============================================================
# 5. Embedding training and cosine similarity
# ============================================================

def test_embedding_training(small_graph):
    trainer = Node2VecTrainer(
        dimensions=8,
        walk_length=12,
        walks_per_node=8,
        window_size=3,
        epochs=3,
        seed=3888,
        workers=1,
    )

    trainer.fit(small_graph)
    embeddings = trainer.get_embeddings()

    assert embeddings.shape == (4, 8)
    assert np.isfinite(embeddings.to_numpy()).all()

    similarities = trainer.similarity_scores("A")

    assert similarities["A"] == pytest.approx(1.0, abs=1e-6)
    assert similarities.between(-1.000001, 1.000001).all()


# ============================================================
# 6. Mean-rank consensus
# ============================================================


def test_consensus_mean_ranks(monkeypatch):
    """
    Verify multi-seed consensus ranking without training Word2Vec.

    Expected ranks:

             B   C   D
    seed 1   1   2   3
    seed 2   3   1   2
    seed 3   1   2   3

    Mean:   5/3 5/3 8/3
    """

    similarities = {
        1: {"B": 0.9, "C": 0.8, "D": 0.7},
        2: {"B": 0.7, "C": 0.9, "D": 0.8},
        3: {"B": 0.9, "C": 0.8, "D": 0.7},
    }

    # --------------------------------------------------------
    # Mock Gensim WordVectors
    # --------------------------------------------------------

    class FakeWordVectors:

        def __init__(self, seed):
            self.scores = similarities[seed]

        def __len__(self):
            # Three candidates + target H
            return len(self.scores) + 1

        def most_similar(self, positive, topn):
            assert positive == ["H"]

            ordered = sorted(
                self.scores.items(),
                key=lambda item: (-item[1], item[0]),
            )

            return ordered[:topn]

    # --------------------------------------------------------
    # Mock Word2Vec model
    # --------------------------------------------------------

    class FakeModel:

        def __init__(self, seed):
            self.wv = FakeWordVectors(seed)

    # --------------------------------------------------------
    # Mock Node2VecTrainer
    # --------------------------------------------------------

    class FakeTrainer:

        def __init__(self, seed):
            self.seed = seed
            self.model = FakeModel(seed)

        @classmethod
        def from_config(cls, config, seed):
            return cls(seed)

        def fit(self, graph):
            return self

    monkeypatch.setattr(
        node2vec,
        "Node2VecTrainer",
        FakeTrainer,
    )

    # --------------------------------------------------------
    # Run consensus
    # --------------------------------------------------------

    G = nx.Graph()
    G.add_nodes_from(["H", "B", "C", "D"])

    result = node2vec.fit_node2vec_consensus(
        G,
        target_id="H",
        config={"training_seeds": [1, 2, 3]},
    )

    # --------------------------------------------------------
    # Verify individual ranks
    # --------------------------------------------------------

    assert result.loc["B", "rank_1"] == 1
    assert result.loc["B", "rank_2"] == 3
    assert result.loc["B", "rank_3"] == 1

    assert result.loc["C", "rank_1"] == 2
    assert result.loc["C", "rank_2"] == 1
    assert result.loc["C", "rank_3"] == 2

    # --------------------------------------------------------
    # Verify mean ranks
    # --------------------------------------------------------

    assert result.loc["B", "mean_rank"] == pytest.approx(5/3)
    assert result.loc["C", "mean_rank"] == pytest.approx(5/3)
    assert result.loc["D", "mean_rank"] == pytest.approx(8/3)

    # --------------------------------------------------------
    # Verify percentile normalisation
    # --------------------------------------------------------

    assert result.loc["B", "N"] == pytest.approx(0.75)
    assert result.loc["C", "N"] == pytest.approx(0.75)
    assert result.loc["D", "N"] == pytest.approx(0.0)

    assert result["N"].between(0, 1).all()


# ============================================================
# 7. Invalid parameters and inputs
# ============================================================

def test_invalid_parameters():
    with pytest.raises(ValueError):
        Node2VecTrainer(p=0)

    with pytest.raises(ValueError):
        Node2VecTrainer(q=-1)

    with pytest.raises(ValueError):
        Node2VecTrainer(dimensions=0)

    trainer = Node2VecTrainer()
    with pytest.raises(RuntimeError):
        trainer.get_embeddings()


def test_invalid_edge_weights():
    G = nx.Graph()
    G.add_edge("A", "B", confidence=-0.5)
    trainer = Node2VecTrainer()
    with pytest.raises(ValueError):
        trainer.generate_walks(G)