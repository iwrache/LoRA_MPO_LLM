"""Tests for mpo_modules.factorization."""

import pytest
import torch
import torch.nn as nn

from mpo_modules.factorization import (
    estimate_mpo_bond_dim,
    factor_linear_mpo,
    get_mpo_compression_ratio,
    robust_svd_split,
)


class TestRobustSvdSplit:
    def test_reconstruction(self):
        torch.manual_seed(0)
        A = torch.randn(32, 16)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        U_bal, T_next = robust_svd_split(U, S, Vh)
        reconstructed = U_bal @ T_next
        original = U @ torch.diag(S) @ Vh
        torch.testing.assert_close(reconstructed, original, atol=1e-5, rtol=1e-5)

    def test_output_shapes(self):
        U = torch.randn(32, 10)
        S = torch.abs(torch.randn(10))
        Vh = torch.randn(10, 16)
        U_bal, T_next = robust_svd_split(U, S, Vh)
        assert U_bal.shape == (32, 10)
        assert T_next.shape == (10, 16)


class TestGetMpoCompressionRatio:
    def test_returns_float_in_range(self):
        ratio = get_mpo_compression_ratio(4096, 4096, 3, 64)
        assert isinstance(ratio, float)
        assert 0 < ratio < 1

    def test_higher_bond_dim_higher_ratio(self):
        r1 = get_mpo_compression_ratio(4096, 4096, 3, 32)
        r2 = get_mpo_compression_ratio(4096, 4096, 3, 128)
        assert r2 > r1

    def test_2_cores(self):
        ratio = get_mpo_compression_ratio(64, 64, 2, 8)
        expected = (64 * 8 + 8 * 64) / (64 * 64)
        assert abs(ratio - expected) < 1e-6


class TestEstimateMpoBondDim:
    def test_positive_result(self):
        bd = estimate_mpo_bond_dim(4096, 4096, 3, target_ratio=0.4)
        assert bd >= 1

    def test_higher_ratio_higher_bond(self):
        bd1 = estimate_mpo_bond_dim(4096, 4096, 3, target_ratio=0.2)
        bd2 = estimate_mpo_bond_dim(4096, 4096, 3, target_ratio=0.8)
        assert bd2 >= bd1

    def test_2_cores(self):
        bd = estimate_mpo_bond_dim(64, 64, 2, target_ratio=0.5)
        assert bd >= 1


class TestFactorLinearMpo:
    def test_basic_shapes(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=8, num_cores=3)
        assert mpo.in_f == 64
        assert mpo.out_f == 64
        assert mpo.num_cores == 3

    def test_forward_shape(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=16, num_cores=3)
        x = torch.randn(4, 64)
        y = mpo(x)
        assert y.shape == (4, 64)

    def test_approximation_quality(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=64, num_cores=2)
        x = torch.randn(8, 64)
        with torch.no_grad():
            y_orig = linear(x)
            y_mpo = mpo(x)
        rel_err = (y_orig - y_mpo).norm() / y_orig.norm()
        assert rel_err < 0.5, f"Approximation error too large: {rel_err:.4f}"

    def test_num_cores_2(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=16, num_cores=2)
        assert mpo.num_cores == 2

    def test_raises_for_1_core(self):
        linear = nn.Linear(64, 64, bias=False)
        with pytest.raises(ValueError, match="num_cores must be >= 2"):
            factor_linear_mpo(linear, bond_dim=8, num_cores=1)

    def test_rectangular(self):
        torch.manual_seed(0)
        linear = nn.Linear(32, 48, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=8, num_cores=2)
        x = torch.randn(4, 32)
        y = mpo(x)
        assert y.shape == (4, 48)
