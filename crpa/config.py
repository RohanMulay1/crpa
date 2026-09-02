"""
crpa.config - typed, immutable configuration objects.

Replaces the mutable global ``CFG`` dict used by the original implementation.
The legacy dict is still produced by :func:`to_legacy_cfg` and consumed by
:func:`from_legacy_cfg`, so ``python main.py`` keeps working unchanged.

Configuration profiles live in ``configs/*.yaml``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

# Canonical variant names, and the aliases that resolve onto them.
CANONICAL_VARIANTS = ("dense", "sliding", "crpa_noreg", "crpa_naive", "crpa_contribution")

VARIANT_ALIASES = {
    # Historical name. Preserved for checkpoint and CLI compatibility.
    "crpa_causal": "crpa_contribution",
}

VARIANT_LABELS = {
    "dense": "Dense Transformer",
    "sliding": "Sliding Window",
    "crpa_noreg": "CRPA no reg.",
    "crpa_naive": "CRPA naive reg.",
    "crpa_contribution": "CRPA contribution-gated",
}

#: The three variants that form the central Tier 1 comparison.
CENTRAL_VARIANTS = ("crpa_noreg", "crpa_naive", "crpa_contribution")


def resolve_variant(name: str) -> str:
    """Map any accepted variant spelling onto its canonical name.

    ``resolve_variant("crpa_causal")`` returns ``"crpa_contribution"``.
    """
    canonical = VARIANT_ALIASES.get(name, name)
    if canonical not in CANONICAL_VARIANTS:
        raise ValueError(
            "unknown variant {!r}; expected one of {} (aliases: {})".format(
                name, CANONICAL_VARIANTS, tuple(VARIANT_ALIASES)
            )
        )
    return canonical


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. Frozen so it can be hashed into a run id."""

    n_embd: int = 192
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.10
    vocab_size: int = 50257
    block_size: int = 512

    # "learned" absolute embeddings (small profile, backwards compatible) or
    # "rope" (long-context profiles - a learned table at 64k would add ~50M
    # parameters and make the advertised parameter count context-dependent).
    position: str = "learned"

    # CRPA structure. Omega(i) = P(i) u G u C_k(i)
    partition_size: int = 128
    n_relays: int = 4
    cross_k: int = 4
    route_temp: float = 0.70

    # Fraction of attention mass defining a row's top-p support.
    overlap_rho: float = 0.60

    # "dense_masked"  - materialises (B,H,T,T); exact, used at short context.
    # "sparse_gather" - block/relay/cross gather; required above ~8k tokens.
    attention_impl: str = "dense_masked"

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd ({}) must be divisible by n_head ({})".format(self.n_embd, self.n_head)
            )
        if self.position not in ("learned", "rope"):
            raise ValueError("position must be 'learned' or 'rope', got {!r}".format(self.position))
        if self.attention_impl not in ("dense_masked", "sparse_gather"):
            raise ValueError(
                "attention_impl must be 'dense_masked' or 'sparse_gather', got {!r}".format(
                    self.attention_impl
                )
            )

    @property
    def d_head(self) -> int:
        return self.n_embd // self.n_head

    def n_params(self) -> int:
        """Analytic parameter count (tied embeddings, biasless projections).

        Excludes the CRPA router, which is reported separately because it is
        absent from the ``dense`` and ``sliding`` baselines.
        """
        emb = self.vocab_size * self.n_embd
        pos = self.block_size * self.n_embd if self.position == "learned" else 0
        attn = 4 * self.n_embd * self.n_embd
        # Two Linear layers with bias: (d -> 4d) and (4d -> d).
        ffn = 8 * self.n_embd * self.n_embd + 4 * self.n_embd + self.n_embd
        ln = 4 * self.n_embd  # two LayerNorms, weight + bias
        per_layer = attn + ffn + ln
        return emb + pos + self.n_layer * per_layer + 2 * self.n_embd


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and the co-training schedule."""

    batch_size: int = 16
    max_iters: int = 4000
    eval_iters: int = 20
    eval_interval: int = 400
    lr: float = 3e-4
    weight_decay: float = 0.10
    grad_clip: float = 1.0
    warmup_steps: int = 200

    # Fraction of steps spent on the retrieval task vs language modelling.
    ret_ratio: float = 0.90

    lambda_bal: float = 0.01
    lambda_red: float = 0.05

    seed: int = 42
    checkpoint_every: int = 1000


@dataclass(frozen=True)
class ContributionConfig:
    """Behavioral-contribution estimation.

    ``mode`` selects what an intervention actually removes:

    ``edge``
        The repaired semantics. A candidate is an edge (layer, head, query i,
        key j) that is *present* in the mask; the intervention zeroes exactly
        that entry. Structural overlap and behavioral contribution then refer
        to the same object.

    ``legacy_rowpair``
        The original repository behaviour, preserved so previously committed
        results stay reproducible. Candidates are *pairs of query rows* scored
        by support Jaccard, but the intervention masks the edge (i -> j).
        Because the mask is causal, every sampled pair with i < j is a no-op
        whose delta is identically zero. Retained for comparison only.
    """

    eps: float = 0.03
    interval: int = 200
    n_pairs: int = 8
    warmup_steps: int = 600
    mode: str = "edge"
    overlap_threshold: float = 0.20

    # Number of intervention samples used when estimating a ranking.
    n_samples: int = 8

    # Group thresholds for the high-overlap analyses (Tier 1 item 4),
    # expressed as quantiles of the measured distributions.
    high_overlap_q: float = 0.75
    low_contribution_q: float = 0.25
    high_contribution_q: float = 0.75

    def __post_init__(self) -> None:
        if self.mode not in ("edge", "legacy_rowpair"):
            raise ValueError(
                "intervention mode must be 'edge' or 'legacy_rowpair', got {!r}".format(self.mode)
            )


@dataclass(frozen=True)
class DataConfig:
    """Needle-in-Haystack construction and the split protocol.

    Three roles are kept strictly disjoint (see :mod:`crpa.data`):
      train        - fits parameters
      calibration  - selects thresholds and estimates contribution
      evaluation   - reports final numbers, touched by nothing else
    """

    filler_range: Tuple[int, int] = (2000, 9999)
    key_range: Tuple[int, int] = (100, 119)
    val_range: Tuple[int, int] = (120, 139)
    needle_depth_range: Tuple[float, float] = (0.55, 0.73)
    needle_depths: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_needles: int = 2

    # True places the query key at the scored position (block_size - 1).
    # False reproduces the original construction, which left the query
    # key two positions from the end and scored the model on a filler
    # token. Kept so the effect of that fix can be attributed rather
    # than inferred.
    query_key_at_end: bool = True

    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    @property
    def uniform_chance(self) -> float:
        """Accuracy of guessing uniformly over the value range, in percent.

        With the default 20-token value range this is 5.0%. **This is a lower
        bound and on its own it is misleading.** Only ``n_needles`` value
        tokens appear in any sequence, so a model that has learned nothing but
        "the answer is a value-range token I can see" already scores about
        ``100 / n_needles``, which is roughly 52% at the default settings.

        Use :meth:`crpa.data.NeedleGenerator.measure_chance_floor`, which
        simulates the generator, for the floor a result must actually clear.
        """
        n_vals = self.val_range[1] - self.val_range[0] + 1
        return 100.0 / n_vals

    @property
    def chance_accuracy(self) -> float:
        """Deprecated alias for :attr:`uniform_chance`.

        Retained so older code keeps running. Do not use it to decide whether a
        result beat chance: it reports 5.0% where the measured floor is about
        52%, which inverts the conclusion.
        """
        return self.uniform_chance


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce one run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    contribution: ContributionConfig = field(default_factory=ContributionConfig)
    data: DataConfig = field(default_factory=DataConfig)

    variant: str = "crpa_contribution"
    profile: str = "small_12m"
    multi_seeds: Tuple[int, ...] = (42, 1337, 2024)
    scale_lens: Tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", resolve_variant(self.variant))

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        """Stable JSON used to derive deterministic run ids."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        sub = {
            "model": ModelConfig,
            "train": TrainConfig,
            "contribution": ContributionConfig,
            "data": DataConfig,
        }
        for key, klass in sub.items():
            if key in d and isinstance(d[key], dict):
                fields = {f.name for f in dataclasses.fields(klass)}
                unknown = set(d[key]) - fields
                if unknown:
                    raise ValueError("unknown {} config keys: {}".format(key, sorted(unknown)))
                payload = dict(d[key])
                for f in dataclasses.fields(klass):
                    if f.name in payload and isinstance(payload[f.name], list):
                        payload[f.name] = tuple(payload[f.name])
                d[key] = klass(**payload)
        for key in ("multi_seeds", "scale_lens"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError("unknown experiment config keys: {}".format(sorted(unknown)))
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("config profile not found: {}".format(path))
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def replace(self, **kwargs: Any) -> "ExperimentConfig":
        """Return a copy with fields replaced.

        Nested fields use dotted keys, e.g. ``cfg.replace(**{"train.lambda_red": 0.1})``.
        """
        nested: Dict[str, Dict[str, Any]] = {}
        top: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if "." in key:
                head, tail = key.split(".", 1)
                nested.setdefault(head, {})[tail] = value
            else:
                top[key] = value
        for head, updates in nested.items():
            top[head] = dataclasses.replace(getattr(self, head), **updates)
        return dataclasses.replace(self, **top)


# ---------------------------------------------------------------------------
#  Legacy bridge
# ---------------------------------------------------------------------------

def to_legacy_cfg(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Render an :class:`ExperimentConfig` as the original flat ``CFG`` dict."""
    return dict(
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        dropout=cfg.model.dropout,
        partition_size=cfg.model.partition_size,
        n_relays=cfg.model.n_relays,
        cross_k=cfg.model.cross_k,
        route_temp=cfg.model.route_temp,
        lambda_bal=cfg.train.lambda_bal,
        lambda_red=cfg.train.lambda_red,
        overlap_rho=cfg.model.overlap_rho,
        sens_eps=cfg.contribution.eps,
        sens_interval=cfg.contribution.interval,
        sens_n_pairs=cfg.contribution.n_pairs,
        batch_size=cfg.train.batch_size,
        max_iters=cfg.train.max_iters,
        eval_iters=cfg.train.eval_iters,
        eval_interval=cfg.train.eval_interval,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.grad_clip,
        warmup_steps=cfg.train.warmup_steps,
        ret_ratio=cfg.train.ret_ratio,
        seed=cfg.train.seed,
        ablation_block_size=cfg.model.block_size,
        scale_lens=list(cfg.scale_lens),
        needle_depths=list(cfg.data.needle_depths),
        multi_seeds=list(cfg.multi_seeds),
        filler_range=cfg.data.filler_range,
        key_range=cfg.data.key_range,
        val_range=cfg.data.val_range,
    )


def from_legacy_cfg(
    legacy: Dict[str, Any],
    variant: str = "crpa_contribution",
    **overrides: Any,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from the original flat ``CFG`` dict."""
    model = ModelConfig(
        n_embd=legacy["n_embd"],
        n_head=legacy["n_head"],
        n_layer=legacy["n_layer"],
        dropout=legacy["dropout"],
        block_size=legacy.get("ablation_block_size", 512),
        partition_size=legacy["partition_size"],
        n_relays=legacy["n_relays"],
        cross_k=legacy["cross_k"],
        route_temp=legacy["route_temp"],
        overlap_rho=legacy["overlap_rho"],
    )
    train = TrainConfig(
        batch_size=legacy["batch_size"],
        max_iters=legacy["max_iters"],
        eval_iters=legacy["eval_iters"],
        eval_interval=legacy["eval_interval"],
        lr=legacy["lr"],
        weight_decay=legacy["weight_decay"],
        grad_clip=legacy["grad_clip"],
        warmup_steps=legacy["warmup_steps"],
        ret_ratio=legacy["ret_ratio"],
        lambda_bal=legacy["lambda_bal"],
        lambda_red=legacy["lambda_red"],
        seed=legacy["seed"],
    )
    contribution = ContributionConfig(
        eps=legacy["sens_eps"],
        interval=legacy["sens_interval"],
        n_pairs=legacy["sens_n_pairs"],
    )
    data = DataConfig(
        filler_range=tuple(legacy["filler_range"]),
        key_range=tuple(legacy["key_range"]),
        val_range=tuple(legacy["val_range"]),
        needle_depths=tuple(legacy["needle_depths"]),
    )
    cfg = ExperimentConfig(
        model=model,
        train=train,
        contribution=contribution,
        data=data,
        variant=variant,
        multi_seeds=tuple(legacy.get("multi_seeds", (42, 1337, 2024))),
        scale_lens=tuple(legacy.get("scale_lens", (64, 128, 256, 512))),
    )
    return cfg.replace(**overrides) if overrides else cfg


def load_profile(name_or_path: str) -> ExperimentConfig:
    """Load ``configs/<name>.yaml``, or a direct path to a YAML file."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return ExperimentConfig.from_yaml(path)
    candidate = Path(__file__).resolve().parent.parent / "configs" / "{}.yaml".format(name_or_path)
    return ExperimentConfig.from_yaml(candidate)
