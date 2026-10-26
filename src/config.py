"""
Configuration management.
All relative paths are resolved from the repository root.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_KEYS = ("models", "rankings", "evaluation", "figures")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a reproducible experiment."""
    root: Path
    config_file: Path
    project: dict[str, Any]
    paths: dict[str, Path]
    network: dict[str, Any]
    candidates: dict[str, Any]
    node2vec: dict[str, Any]
    rwr: dict[str, Any]
    louvain: dict[str, Any]
    cadtp: dict[str, Any]
    evaluation: dict[str, Any]
    visualisation: dict[str, Any]

    def path(self, name: str) -> Path:
        """Return an absolute path defined in the YAML."""

        if name not in self.paths:
            raise KeyError(f"Unknown path: {name}")

        return self.paths[name]

    def ensure_output_dirs(self) -> None:
        """Create output directories if necessary."""

        for name in OUTPUT_KEYS:
            self.path(name).mkdir(parents=True, exist_ok=True)

    def check_inputs(self, *names: str) -> None:
        """Check that specified input files exist."""

        missing = [
            str(self.path(name))
            for name in names
            if not self.path(name).is_file()
        ]

        if missing:
            raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))


def validate_config(cfg: ExperimentConfig) -> None:
    """Validate essential experimental parameters."""

    # Network
    threshold = cfg.network["confidence_threshold"]
    divisor = cfg.network["confidence_divisor"]

    if not 0 <= threshold <= 1000:
        raise ValueError("Network threshold must be between 0 and 1000.")

    if divisor <= 0:
        raise ValueError("Confidence divisor must be positive.")

    # Node2Vec
    n2v = cfg.node2vec

    for key in (
        "dimensions",
        "walk_length",
        "walks_per_node",
        "window_size",
        "epochs"):
        if n2v[key] <= 0:
            raise ValueError(f"node2vec.{key} must be positive.")

    if n2v["p"] <= 0 or n2v["q"] <= 0:
        raise ValueError("Node2Vec p and q must be positive.")

    if not n2v["training_seeds"]:
        raise ValueError("Node2Vec requires training seeds.")

    # Random Walk with Restart
    if not 0 < cfg.rwr["alpha"] < 1:
        raise ValueError("RWR alpha must be between 0 and 1.")

    # Louvain
    train = cfg.louvain["training_seeds"]
    validation = cfg.louvain["validation_seeds"]

    if not train or not validation:
        raise ValueError("Louvain seed lists cannot be empty.")

    if len(set(train)) != len(train):
        raise ValueError("Duplicate Louvain training seeds.")

    if len(set(validation)) != len(validation):
        raise ValueError("Duplicate Louvain validation seeds.")

    if set(train) & set(validation):
        raise ValueError("Training and validation Louvain seeds must be disjoint.")

    # CADTP
    local = cfg.cadtp["local_threshold"]
    cross = cfg.cadtp["cross_threshold"]

    if not 0 <= cross < local <= 1:
        raise ValueError(
            "CADTP thresholds must satisfy "
            "0 <= cross < local <= 1."
        )

    if cfg.cadtp["affinity_smoothing"] < 0:
        raise ValueError("Community affinity smoothing must be non-negative.")

    for key in (
        "local_weights",
        "cross_weights",
        "boundary_weights"):
        weights = cfg.cadtp[key]

        if any(weight < 0 for weight in weights.values()):
            raise ValueError(f"{key} contains negative weights.")

        if sum(weights.values()) <= 0:
            raise ValueError(f"{key} must have positive total weight.")


def load_config(
    file: str | Path = "configs/default.yaml",
    root: Path | None = None,
) -> ExperimentConfig:
    """
    Load and validate an experiment configuration.

    Parameters
    ----------
    file : str or Path
        YAML configuration file.
    root : Path, optional
        Repository root. Defaults to the directory
        containing src/.

    Returns
    -------
    ExperimentConfig
        Validated configuration with absolute paths.
    """

    repo_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else ROOT
    )

    config_file = Path(file).expanduser()

    if not config_file.is_absolute():
        config_file = repo_root / config_file

    config_file = config_file.resolve()

    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    sections = [
        "project",
        "paths",
        "network",
        "candidates",
        "node2vec",
        "rwr",
        "louvain",
        "cadtp",
        "evaluation",
        "visualisation",
    ]

    for section in sections:
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"Missing or invalid section: {section}")

    # Resolve relative paths against repository root
    resolved_paths = {}

    for name, value in data["paths"].items():
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = repo_root / path

        resolved_paths[name] = path.resolve()

    cfg = ExperimentConfig(
        root=repo_root,
        config_file=config_file,
        project=data["project"],
        paths=resolved_paths,
        network=data["network"],
        candidates=data["candidates"],
        node2vec=data["node2vec"],
        rwr=data["rwr"],
        louvain=data["louvain"],
        cadtp=data["cadtp"],
        evaluation=data["evaluation"],
        visualisation=data["visualisation"],
    )

    validate_config(cfg)

    return cfg


if __name__ == "__main__":
    cfg = load_config()

    print("Configuration loaded successfully!")
    print(f"Repository: {cfg.root}")
    print(f"Target: {cfg.project['target_protein']}")
    print(f"STRING threshold: {cfg.network['confidence_threshold']}")
    print(f"Node2Vec dimensions: {cfg.node2vec['dimensions']}")
    print(f"Louvain training runs: {len(cfg.louvain['training_seeds'])}")
    print(f"Louvain validation runs: {len(cfg.louvain['validation_seeds'])}")
