"""Shared fixtures. Everything here runs on CPU in seconds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crpa.config import (
    ContributionConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from crpa.data import Corpus
from crpa.model import GPT

TINY_VOCAB = 10_000


@pytest.fixture(scope="session")
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        n_embd=32, n_head=4, n_layer=2, dropout=0.0,
        vocab_size=TINY_VOCAB, block_size=64,
        partition_size=16, n_relays=2, cross_k=2,
    )


@pytest.fixture(scope="session")
def tiny_config(tiny_model_config: ModelConfig) -> ExperimentConfig:
    return ExperimentConfig(
        model=tiny_model_config,
        train=TrainConfig(batch_size=2, max_iters=4, eval_iters=1,
                          eval_interval=2, warmup_steps=1, checkpoint_every=0),
        contribution=ContributionConfig(warmup_steps=1, interval=2, n_pairs=3),
        data=DataConfig(),
        variant="crpa_contribution",
    )


@pytest.fixture(scope="session")
def corpus(tiny_config: ExperimentConfig) -> Corpus:
    return Corpus(tiny_config.data, seed=42).load_synthetic(
        vocab_size=TINY_VOCAB, n_tokens=20_000
    )


@pytest.fixture
def tiny_model(tiny_model_config: ModelConfig) -> GPT:
    torch.manual_seed(0)
    model = GPT(tiny_model_config, "crpa_contribution", seed=7)
    model.eval()
    return model


@pytest.fixture
def needle_batch(corpus: Corpus, tiny_model_config: ModelConfig):
    return corpus.needle_batch("calibration", tiny_model_config.block_size, 2)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
