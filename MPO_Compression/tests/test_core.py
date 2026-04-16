"""Tests for mpo_modules.core (MPOLinear)."""

import torch

from mpo_modules.core import MPOLinear


class TestMPOLinearInit:
    def test_basic_attributes(self, random_mpo_cores_3):
        mpo = MPOLinear(64, 64, random_mpo_cores_3)
        assert mpo.in_f == 64
        assert mpo.out_f == 64
        assert mpo.num_cores == 3
        assert len(mpo.cores) == 3

    def test_2_cores(self, random_mpo_cores_2):
        mpo = MPOLinear(64, 64, random_mpo_cores_2)
        assert mpo.num_cores == 2


class TestMPOLinearForward:
    def test_output_shape(self, random_mpo_cores_3):
        mpo = MPOLinear(64, 64, random_mpo_cores_3)
        x = torch.randn(4, 64)
        y = mpo(x)
        assert y.shape == (4, 64)

    def test_output_finite(self, random_mpo_cores_3):
        mpo = MPOLinear(64, 64, random_mpo_cores_3)
        x = torch.randn(4, 64)
        y = mpo(x)
        assert torch.isfinite(y).all()

    def test_batch_dim_preserved(self, random_mpo_cores_3):
        mpo = MPOLinear(64, 64, random_mpo_cores_3)
        for bs in [1, 8, 32]:
            x = torch.randn(bs, 64)
            y = mpo(x)
            assert y.shape[0] == bs


class TestMPOLinearWeightReconstruction:
    def test_shape(self, random_mpo_cores_3):
        mpo = MPOLinear(64, 64, random_mpo_cores_3)
        W = mpo._build_full_weight_fp32()
        assert W.shape == (64, 64)

    def test_finite(self, random_mpo_cores_3):
        mpo = MPOLinear(64, 64, random_mpo_cores_3)
        W = mpo._build_full_weight_fp32()
        assert torch.isfinite(W).all()


class TestCleanFinite:
    def test_clean_tensor_unchanged(self):
        x = torch.randn(10)
        y = MPOLinear._clean_finite(x)
        torch.testing.assert_close(x, y)

    def test_nan_replaced(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        y = MPOLinear._clean_finite(x)
        assert torch.isfinite(y).all()

    def test_inf_replaced(self):
        x = torch.tensor([1.0, float("inf"), -float("inf")])
        y = MPOLinear._clean_finite(x)
        assert torch.isfinite(y).all()
