"""
Weighted Node2Vec implementation.
Features:
    - Second-order biased random walks
    - STRING confidence-weighted transitions
    - Skip-gram Word2Vec embeddings
    - Cosine similarity to a target protein
    - Multi-seed consensus ranking

Graph nodes must be string identifiers.
"""
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from .algorithms import rank_percentile


# ============================================================
# 1. Node2Vec trainer
# ============================================================
class Node2VecTrainer:
    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 40,
        walks_per_node: int = 5,
        window_size: int = 5,
        p: float = 1.0,
        q: float = 1.0,
        negative_samples: int = 5,
        epochs: int = 5,
        seed: int = 3888,
        workers: int = 1,
        weight: str = "confidence",
    ):
        if min(
            dimensions,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            workers) <= 0:
            raise ValueError("Training dimensions and counts must be positive.")

        if p <= 0 or q <= 0:
            raise ValueError("Node2Vec p and q must be positive.")

        if negative_samples < 0:
            raise ValueError("negative_samples cannot be negative.")

        self.dimensions = dimensions
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.window_size = window_size

        self.p = float(p)
        self.q = float(q)

        self.negative_samples = negative_samples
        self.epochs = epochs
        self.seed = int(seed)
        self.workers = workers
        self.weight = weight

        self.model = None
        self.nodes = None

        self._neighbors = None
        self._neighbor_sets = None
        self._weights = None
        self._cumulative_weights = None

    # --------------------------------------------------------
    # 2. Initialise graph data
    # --------------------------------------------------------
    def _prepare_graph(self, graph: nx.Graph) -> None:
        if graph.is_directed():
            raise ValueError(
                "This Node2Vec implementation expects "
                "an undirected graph."
            )

        if not all(isinstance(v, str) for v in graph.nodes()):
            raise TypeError("All graph node identifiers must be strings.")

        self.nodes = sorted(graph.nodes())

        self._neighbors = {}
        self._neighbor_sets = {}
        self._weights = {}
        self._cumulative_weights = {}

        for v in self.nodes:
            neighbors = sorted(graph.neighbors(v))
            weights = np.array([
                float(graph[v][u].get(self.weight, 1.0))
                for u in neighbors
            ], dtype=float)

            if (not np.isfinite(weights).all()
                or (weights <= 0).any()):
                raise ValueError(f"Invalid edge weights at node {v}.")

            self._neighbors[v] = neighbors
            self._neighbor_sets[v] = set(neighbors)
            self._weights[v] = weights

            if len(weights):
                cdf = np.cumsum(weights)
                cdf /= cdf[-1]
                cdf[-1] = 1.0
            else:
                cdf = np.array([], dtype=float)

            self._cumulative_weights[v] = cdf

    # --------------------------------------------------------
    # 3. Sample one transition
    # --------------------------------------------------------

    @staticmethod
    def _weighted_choice(
        neighbors: list[str],
        cumulative: np.ndarray,
        rng: np.random.Generator,
    ) -> str:
        draw = rng.random()
        index = int(
            np.searchsorted(
                cumulative,
                draw,
                side="right",
            )
        )

        return neighbors[min(index, len(neighbors) - 1)]

    def _sample_next(
        self,
        previous: str | None,
        current: str,
        rng: np.random.Generator,
    ) -> str | None:
        neighbors = self._neighbors[current]

        if not neighbors:
            return None

        # First step or unbiased case
        if previous is None or (self.p == 1.0 and self.q == 1.0):
            return self._weighted_choice(
                neighbors,
                self._cumulative_weights[current],
                rng,
            )

        # Second-order Node2Vec transition biases
        previous_neighbors = self._neighbor_sets[previous]

        bias = np.array([
            1.0 / self.p
            if x == previous
            else (1.0 if x in previous_neighbors else 1.0 / self.q)
            for x in neighbors
        ], dtype=float)

        adjusted_weights = self._weights[current] * bias
        cumulative = np.cumsum(adjusted_weights)
        cumulative /= cumulative[-1]
        cumulative[-1] = 1.0

        return self._weighted_choice(neighbors, cumulative, rng)

    # --------------------------------------------------------
    # 4. Generate a random walk
    # --------------------------------------------------------

    def _random_walk(self, start: str, rng: np.random.Generator) -> list[str]:
        walk = [start]
        while len(walk) < self.walk_length:
            current = walk[-1]
            previous = walk[-2] if len(walk) >= 2 else None
            next_node = self._sample_next(previous, current, rng)
            if next_node is None:
                break
            walk.append(next_node)

        return walk

    # --------------------------------------------------------
    # 5. Generate all walks
    # --------------------------------------------------------

    def generate_walks(
        self,
        graph: nx.Graph,
    ) -> list[list[str]]:
        """
        Generate walks with a shuffled node order
        in each round.
        """
        self._prepare_graph(graph)
        rng = np.random.default_rng(self.seed)
        walks = []
        for _ in range(self.walks_per_node):
            for idx in rng.permutation(len(self.nodes)):
                node = self.nodes[int(idx)]
                walk = self._random_walk(str(node), rng)
                walks.append(walk)

        return walks

    # --------------------------------------------------------
    # 6. Train embeddings
    # --------------------------------------------------------

    def fit(self, graph: nx.Graph):
        """
        Train skip-gram embeddings from Node2Vec walks.
        """
        walks = self.generate_walks(graph)
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=1,
            sg=1,
            hs=0,
            negative=self.negative_samples,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            sample=0,
            sorted_vocab=1,
        )

        return self

    # --------------------------------------------------------
    # 7. Retrieve embeddings
    # --------------------------------------------------------

    def get_embeddings(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Node2Vec model has not been trained.")

        vectors = np.vstack([self.model.wv[v] for v in self.nodes])
        return pd.DataFrame(vectors, index=pd.Index(self.nodes, name="string_id"))

    # --------------------------------------------------------
    # 8. Cosine similarity to target
    # --------------------------------------------------------

    def similarity_scores(self, target_id: str) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Node2Vec model has not been trained.")

        if target_id not in self.model.wv:
            raise ValueError(f"Target not in embeddings: {target_id}")

        vectors = self.get_embeddings().to_numpy()
        target_vector = self.model.wv[target_id]
        denominator = (
            np.linalg.norm(vectors, axis=1)
            * np.linalg.norm(target_vector)
        )

        if (denominator <= 0).any():
            raise ValueError(
                "Cannot compute cosine similarity "
                "for zero-length embeddings."
            )

        scores = vectors @ target_vector / denominator

        return pd.Series(
            scores,
            index=pd.Index(self.nodes, name="string_id"),
            name="node2vec_cosine",
        )

    # --------------------------------------------------------
    # 9. Save trained model
    # --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))

    # --------------------------------------------------------
    # 10. Construct from YAML section
    # --------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        seed: int,
    ):
        return cls(
            dimensions=config["dimensions"],
            walk_length=config["walk_length"],
            walks_per_node=config["walks_per_node"],
            window_size=config["window_size"],
            p=config["p"],
            q=config["q"],
            negative_samples=config["negative_samples"],
            epochs=config["epochs"],
            seed=seed,
            workers=1,
        )


# ============================================================
# 11. Multi-seed consensus
# ============================================================

def fit_node2vec_consensus(
    graph: nx.Graph,
    target_id: str,
    config: dict[str, Any],
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Train Node2Vec for all configured seeds.

    Ranking procedure:
        1. Compute cosine similarity to target for each seed.
        2. Rank non-target proteins within each seed.
        3. Average the ranks.
        4. Convert mean rank to rank-percentile N.

    Returns:
        DataFrame containing per-seed similarity,
        per-seed ranks, mean_rank and N.
    """
    seeds = config["training_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Node2Vec seeds must be nonempty and unique.")

    nodes = sorted(v for v in graph.nodes() if v != target_id)
    consensus = pd.DataFrame(index=pd.Index(nodes, name="string_id"))

    rank_columns = []
    for seed in seeds:
        print(f"Training Node2Vec seed={seed}...")
        trainer = Node2VecTrainer.from_config(config, seed=seed)
        trainer.fit(graph)

        # Match the original Notebook 04 ranking procedure
        ordered_similarities = trainer.model.wv.most_similar(
            positive=[target_id],
            topn=len(trainer.model.wv) - 1,
        )

        similarity_map = dict(ordered_similarities)

        rank_map = {
            protein_id: rank
            for rank, (protein_id, _) in enumerate(
                ordered_similarities,
                start=1,
            )
        }

        similarity_col = f"cosine_{seed}"
        rank_col = f"rank_{seed}"

        consensus[similarity_col] = pd.Series(
            similarity_map
        ).reindex(consensus.index)

        consensus[rank_col] = pd.Series(
            rank_map
        ).reindex(consensus.index).astype(int)

        rank_columns.append(rank_col)
        if model_dir is not None:
            trainer.save(Path(model_dir) / f"node2vec_seed_{seed}.model")

    # Mean rank across random seeds
    consensus["mean_rank"] = consensus[rank_columns].mean(axis=1)

    # Smaller rank is better, so negate before
    # applying descending rank-percentile.
    consensus["N"] = rank_percentile(-consensus["mean_rank"])

    return consensus


# ============================================================
# 12. Smoke test with a small graph
# ============================================================

if __name__ == "__main__":
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ("A", "B", 0.9),
            ("B", "C", 0.8),
            ("C", "D", 0.7),
            ("D", "E", 0.9),
            ("E", "A", 0.6),
            ("B", "D", 0.5),
        ],
        weight="confidence",
    )

    trainer = Node2VecTrainer(
        dimensions=8,
        walk_length=10,
        walks_per_node=5,
        window_size=3,
        negative_samples=2,
        epochs=3,
        seed=3888,
    )

    walks = trainer.generate_walks(G)
    assert len(walks) == G.number_of_nodes() * trainer.walks_per_node
    assert all(1 <= len(walk) <= trainer.walk_length
        for walk in walks
    )

    # Every transition must follow a real edge
    assert all(G.has_edge(u, v)
        for walk in walks
        for u, v in zip(walk[:-1], walk[1:])
    )

    trainer.fit(G)

    embeddings = trainer.get_embeddings()
    assert embeddings.shape == G.number_of_nodes(), 8

    scores = trainer.similarity_scores("A")
    assert np.isclose(scores["A"], 1.0, atol=1e-6)
    assert scores.between(-1.000001, 1.000001).all()

    print("Node2Vec smoke test passed!")
    print("\nSimilarity to A:")
    print(scores.sort_values(ascending=False))