"""Tests for mpo_modules.tt_ops."""

import pytest
import torch

from mpo_modules.tt_ops import get_tt_rank_info, matrix_tt_svd, tt_round_4d_cores


class TestMatrixTtSvd:
    def test_shapes_2_cores(self):
        A = torch.randn(64, 64)
        cores = matrix_tt_svd(A, in_factors=[8, 8], max_rank=16)
        assert len(cores) == 2
        assert cores[0].shape[0] == 1  # r_0 = 1
        assert cores[-1].shape[-1] == 1  # r_K = 1
        assert cores[0].shape[1] == 8  # o_0
        assert cores[0].shape[2] == 8  # i_0

    def test_shapes_3_cores(self):
        A = torch.randn(64, 64)
        cores = matrix_tt_svd(A, in_factors=[4, 4, 4], max_rank=16)
        assert len(cores) == 3
        assert cores[0].shape[0] == 1
        assert cores[-1].shape[-1] == 1

    def test_reconstruction_accuracy(self):
        torch.manual_seed(0)
        A = torch.randn(64, 64)
        cores = matrix_tt_svd(A, in_factors=[8, 8], max_rank=64)
        # Reconstruct: contract cores
        W = cores[0]
        for c in cores[1:]:
            W = torch.tensordot(W, c, dims=([-1], [0]))
        # W shape: [1, 8, 8, 8, 8, 1] -> remove boundary dims
        W = W.squeeze(0).squeeze(-1)  # [8, 8, 8, 8]
        # Permute to [o0, o1, i0, i1] -> [o0, i0, o1, i1] -> reshape
        W = W.permute(0, 2, 1, 3).reshape(64, 64)
        rel_err = (A - W).norm() / A.norm()
        assert rel_err < 0.1, f"Reconstruction error too large: {rel_err:.4f}"

    def test_rank_control(self):
        A = torch.randn(64, 64)
        cores = matrix_tt_svd(A, in_factors=[8, 8], max_rank=4)
        # Bond dimension should be <= max_rank
        assert cores[0].shape[-1] <= 4

    def test_asserts_square(self):
        with pytest.raises(AssertionError):
            matrix_tt_svd(torch.randn(32, 64), in_factors=[8, 8])

    def test_asserts_factor_product(self):
        with pytest.raises(AssertionError):
            matrix_tt_svd(torch.randn(64, 64), in_factors=[8, 4])  # 32 != 64


class TestTtRound4dCores:
    def test_reduces_bond_dim(self, random_mpo_cores_3):
        cores = [c.clone() for c in random_mpo_cores_3]
        # Original bond dims are 8
        rounded = tt_round_4d_cores(cores, chi_cap=4)
        assert len(rounded) == 3
        # Check bond dims are reduced
        for i in range(len(rounded) - 1):
            bond = rounded[i].shape[-1]
            assert bond <= 4, f"Core {i} bond dim {bond} > 4"

    def test_preserves_boundary(self, random_mpo_cores_3):
        cores = [c.clone() for c in random_mpo_cores_3]
        rounded = tt_round_4d_cores(cores, chi_cap=4)
        assert rounded[0].shape[0] == 1  # left boundary
        assert rounded[-1].shape[-1] == 1  # right boundary

    def test_output_finite(self, random_mpo_cores_3):
        cores = [c.clone() for c in random_mpo_cores_3]
        rounded = tt_round_4d_cores(cores, chi_cap=4)
        for i, c in enumerate(rounded):
            assert torch.isfinite(c).all(), f"Core {i} has non-finite values"


class TestGetTtRankInfo:
    def test_basic(self, random_mpo_cores_3):
        info = get_tt_rank_info(random_mpo_cores_3)
        assert info["num_cores"] == 3
        assert len(info["ranks"]) == 2  # K-1 internal bonds
        assert info["max_rank"] == 8
        assert len(info["shapes"]) == 3

    def test_empty(self):
        info = get_tt_rank_info([])
        assert info["num_cores"] == 0
        assert info["ranks"] == []
        assert info["max_rank"] == 0

    def test_2_cores(self, random_mpo_cores_2):
        info = get_tt_rank_info(random_mpo_cores_2)
        assert info["num_cores"] == 2
        assert len(info["ranks"]) == 1
        assert info["ranks"][0] == 16
