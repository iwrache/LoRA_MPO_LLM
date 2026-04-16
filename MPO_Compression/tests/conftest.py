"""Shared fixtures for MPO_Compression tests."""

import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (require GPU + model weights)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires GPU or model downloads")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="need --run-integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clean_mpo_env(monkeypatch):
    """Remove all MPO_* environment variables to ensure deterministic tests."""
    for key in list(os.environ):
        if key.startswith("MPO_") or key.startswith("HEAL_"):
            monkeypatch.delenv(key, raising=False)
    # Force dense forward path so tests don't require opt_einsum/cotengra
    monkeypatch.setenv("MPO_EVAL_PATH", "dense")
    monkeypatch.setenv("MPO_TRAIN_PATH", "dense")


@pytest.fixture
def small_weight_square():
    """64x64 random weight matrix, seeded for reproducibility."""
    g = torch.Generator().manual_seed(42)
    return torch.randn(64, 64, generator=g)


@pytest.fixture
def small_weight_rect():
    """48x32 random weight matrix, seeded for reproducibility."""
    g = torch.Generator().manual_seed(42)
    return torch.randn(48, 32, generator=g)


@pytest.fixture
def random_mpo_cores_3():
    """3 valid 4D MPO cores: [1,4,4,8], [8,4,4,8], [8,4,4,1] (in/out=64)."""
    g = torch.Generator().manual_seed(42)
    c0 = torch.randn(1, 4, 4, 8, generator=g) * 0.1
    c1 = torch.randn(8, 4, 4, 8, generator=g) * 0.1
    c2 = torch.randn(8, 4, 4, 1, generator=g) * 0.1
    return [c0, c1, c2]


@pytest.fixture
def random_mpo_cores_2():
    """2 valid 4D MPO cores: [1,8,8,16], [16,8,8,1] (in/out=64)."""
    g = torch.Generator().manual_seed(42)
    c0 = torch.randn(1, 8, 8, 16, generator=g) * 0.1
    c1 = torch.randn(16, 8, 8, 1, generator=g) * 0.1
    return [c0, c1]
