"""Tests for mpo_modules.helpers."""

import math

import pytest
import torch

from mpo_modules.helpers import find_factors_balanced, log_tensor


class TestFindFactorsBalanced:
    @pytest.mark.parametrize(
        "n,k",
        [
            (64, 3),
            (128, 2),
            (4096, 3),
            (11008, 3),
            (256, 4),
            (16, 2),
        ],
    )
    def test_product_correct(self, n, k):
        factors = find_factors_balanced(n, k)
        assert len(factors) == k
        assert math.prod(factors) == n

    def test_single_factor(self):
        assert find_factors_balanced(42, 1) == [42]

    def test_power_of_two(self):
        factors = find_factors_balanced(64, 3)
        assert math.prod(factors) == 64
        assert len(factors) == 3
        # balanced: all factors should be 4
        assert factors == [4, 4, 4]

    def test_balance_ratio(self):
        factors = find_factors_balanced(4096, 3)
        ratio = max(factors) / min(factors)
        assert ratio <= 4  # should be reasonably balanced

    def test_prime_input(self):
        factors = find_factors_balanced(17, 3)
        assert math.prod(factors) == 17
        assert len(factors) == 3

    def test_all_factors_positive(self):
        factors = find_factors_balanced(100, 3)
        assert all(f >= 1 for f in factors)
        assert math.prod(factors) == 100


class TestLogTensor:
    def test_clean_tensor_no_output(self, capsys):
        t = torch.randn(10, 10)
        log_tensor("test", t)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_nan_tensor_prints_warning(self, capsys):
        t = torch.tensor([1.0, float("nan"), 3.0])
        log_tensor("test_nan", t)
        captured = capsys.readouterr()
        assert "NaN=True" in captured.out

    def test_inf_tensor_prints_warning(self, capsys):
        t = torch.tensor([1.0, float("inf"), 3.0])
        log_tensor("test_inf", t)
        captured = capsys.readouterr()
        assert "Inf=True" in captured.out

    def test_raise_on_bad(self):
        t = torch.tensor([float("nan")])
        with pytest.raises(RuntimeError, match="NaN=True"):
            log_tensor("bad", t, raise_on_bad=True)

    def test_non_tensor_noop(self):
        log_tensor("not_tensor", 42)  # should not raise
