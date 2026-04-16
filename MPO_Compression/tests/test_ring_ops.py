"""Tests for mpo_modules.ring_ops -- MPO ring (periodic boundary) decomposition."""

import torch
import torch.nn as nn

from mpo_modules.ring_ops import contract_ring, get_ring_rank_info, matrix_ring_svd


class TestMatrixRingSvd:
    def test_shapes_matching_boundary(self):
        """Ring cores should have matching first/last bond dims."""
        torch.manual_seed(42)
        A = torch.randn(64, 64)
        cores = matrix_ring_svd(A, in_factors=[8, 8], max_rank=16, ring_rank=4)
        assert cores[0].shape[0] == cores[-1].shape[-1]  # periodic bond matches
        assert cores[0].shape[0] == 4  # ring_rank

    def test_shapes_3_cores(self):
        """3-core ring decomposition shapes."""
        torch.manual_seed(42)
        A = torch.randn(64, 64)
        cores = matrix_ring_svd(A, in_factors=[4, 4, 4], max_rank=16, ring_rank=4)
        assert len(cores) == 3
        assert cores[0].shape[0] == 4  # ring bond
        assert cores[-1].shape[-1] == 4  # ring bond


class TestContractRing:
    def test_output_shape(self):
        """contract_ring should produce [n, n] matrix."""
        torch.manual_seed(42)
        cores = [
            torch.randn(4, 8, 8, 8),
            torch.randn(8, 8, 8, 4),
        ]
        W = contract_ring(cores)
        assert W.shape == (64, 64)

    def test_reconstruction_accuracy(self):
        """With high rank, ring should reconstruct well."""
        torch.manual_seed(42)
        A = torch.randn(16, 16)
        cores = matrix_ring_svd(A, in_factors=[4, 4], max_rank=16, ring_rank=16)
        W = contract_ring(cores)
        rel_err = (A - W).norm() / A.norm()
        assert rel_err < 0.15, f"Reconstruction error too large: {rel_err:.4f}"


class TestRingRank1:
    def test_rank1_reconstruction(self):
        """Ring with ring_rank=1 should still approximate the matrix."""
        torch.manual_seed(42)
        A = torch.randn(16, 16)
        cores = matrix_ring_svd(A, in_factors=[4, 4], max_rank=16, ring_rank=1)
        assert cores[0].shape[0] == 1
        assert cores[-1].shape[-1] == 1
        W = contract_ring(cores)
        assert W.shape == (16, 16)
        # Should still be a reasonable approximation
        rel_err = (A - W).norm() / A.norm()
        assert rel_err < 1.0  # loose bound for rank-1


class TestRingGradientFlow:
    def test_gradient_finite(self):
        """Backward through contract_ring should produce finite gradients."""
        cores = [
            torch.randn(4, 4, 4, 8, requires_grad=True),
            torch.randn(8, 4, 4, 4, requires_grad=True),
        ]
        W = contract_ring(cores)
        loss = W.sum()
        loss.backward()
        for i, c in enumerate(cores):
            assert c.grad is not None, f"Core {i} has no gradient"
            assert torch.isfinite(c.grad).all(), f"Core {i} has non-finite gradient"


class TestGetRingRankInfo:
    def test_basic(self):
        cores = [
            torch.randn(4, 8, 8, 8),
            torch.randn(8, 8, 8, 4),
        ]
        info = get_ring_rank_info(cores)
        assert info["num_cores"] == 2
        assert info["ring_bond"] == 4
        assert len(info["ranks"]) == 1
        assert info["ranks"][0] == 8

    def test_empty(self):
        info = get_ring_rank_info([])
        assert info["num_cores"] == 0
        assert info["ring_bond"] == 0


class TestFactorLinearMpoRing:
    def test_boundary_periodic(self):
        """factor_linear_mpo with boundary='periodic' returns periodic MPOLinear."""
        from mpo_modules.factorization import factor_linear_mpo

        torch.manual_seed(42)
        linear = nn.Linear(64, 64, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=8, num_cores=3, boundary="periodic")
        assert mpo.boundary == "periodic"
        assert mpo.cores[0].data.shape[0] == mpo.cores[-1].data.shape[-1]

    def test_ring_forward_shape(self):
        """MPOLinear with periodic boundary produces correct output shape."""
        from mpo_modules.factorization import factor_linear_mpo

        torch.manual_seed(42)
        linear = nn.Linear(64, 64, bias=False)
        mpo = factor_linear_mpo(linear, bond_dim=8, num_cores=2, boundary="periodic")
        x = torch.randn(4, 64)
        y = mpo(x)
        assert y.shape == (4, 64)

    def test_state_dict_round_trip(self):
        """Periodic boundary survives state_dict save/load."""
        from mpo_modules.core import MPOLinear

        torch.manual_seed(42)
        cores = [torch.randn(4, 8, 8, 8), torch.randn(8, 8, 8, 4)]
        mpo = MPOLinear(64, 64, cores, boundary="periodic")
        assert mpo.boundary == "periodic"

        # Save and load into a fresh module with default boundary
        sd = mpo.state_dict()
        mpo2 = MPOLinear(64, 64, cores, boundary="open")
        mpo2.load_state_dict(sd)
        assert mpo2.boundary == "periodic"  # synced from buffer


class TestContractRingOrder:
    def test_io_order_matches_oi_transpose(self):
        """contract_ring with order='io' should produce same W as order='oi'."""
        torch.manual_seed(42)
        A = torch.randn(16, 16)
        cores = matrix_ring_svd(A, in_factors=[4, 4], max_rank=16, ring_rank=4)
        W_oi = contract_ring(cores, order="oi")
        W_io = contract_ring(cores, order="io")
        torch.testing.assert_close(W_oi, W_io, atol=1e-5, rtol=1e-5)


class TestChainToRingDeterminism:
    def test_zero_noise_exact(self):
        """chain_to_ring with default eps_noise=0 preserves exact chain reconstruction."""
        from mpo_modules.ring_ops import chain_to_ring
        from mpo_modules.tt_ops import matrix_tt_svd

        torch.manual_seed(42)
        A = torch.randn(16, 16)
        chain_cores = matrix_tt_svd(A, [4, 4], max_rank=16)
        ring_cores = chain_to_ring(chain_cores, ring_rank=4)
        W = contract_ring(ring_cores)
        # With zero noise, only alpha=0 contributes → exact chain result
        # Reconstruct chain for comparison
        x = chain_cores[0]
        for c in chain_cores[1:]:
            x = torch.tensordot(x, c, dims=([-1], [0]))
        x = x.squeeze(0).squeeze(-1)
        x = x.permute(0, 2, 1, 3).reshape(16, 16)
        torch.testing.assert_close(W, x, atol=1e-5, rtol=1e-5)
