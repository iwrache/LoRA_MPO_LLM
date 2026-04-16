"""Tests for mpo_modules.factorization_utils."""

import math

from mpo_modules.factorization_utils import (
    _find_factors_edge_heavy,
    _reorder_ofac_ifac,
    compute_mpo_params_edge_heavy,
    get_chi_max_for_layer,
)


class TestReorderOfacIfac:
    def test_out_descending_in_ascending(self):
        ofac, ifac = _reorder_ofac_ifac([2, 8, 4], [3, 1, 9])
        assert ofac == sorted(ofac, reverse=True)
        assert ifac == sorted(ifac)

    def test_already_ordered(self):
        ofac, ifac = _reorder_ofac_ifac([8, 4, 2], [1, 3, 9])
        assert ofac == [8, 4, 2]
        assert ifac == [1, 3, 9]

    def test_preserves_products(self):
        ofac, ifac = _reorder_ofac_ifac([3, 5, 2], [7, 4, 6])
        assert math.prod(ofac) == 30
        assert math.prod(ifac) == 168


class TestFindFactorsEdgeHeavy:
    def test_default_balanced(self):
        """Without env vars, should use balanced factoring."""
        ofac, ifac = _find_factors_edge_heavy(64, 64, 3)
        assert math.prod(ofac) == 64
        assert math.prod(ifac) == 64

    def test_product_correct_for_large(self):
        ofac, ifac = _find_factors_edge_heavy(4096, 4096, 3)
        assert math.prod(ofac) == 4096
        assert math.prod(ifac) == 4096

    def test_with_mlp_edge_heavy(self, monkeypatch):
        monkeypatch.setenv("MPO_MLP_EDGE_HEAVY", "1")
        ofac, ifac = _find_factors_edge_heavy(11008, 4096, 3, layer_name="model.layers.0.mlp.up_proj")
        assert math.prod(ofac) == 11008
        assert math.prod(ifac) == 4096


class TestGetChiMaxForLayer:
    def test_3_cores_positive(self):
        result = get_chi_max_for_layer(64, 64, 3)
        assert result > 0

    def test_2_cores_positive(self):
        result = get_chi_max_for_layer(64, 64, 2)
        assert result > 0


class TestComputeMpoParamsEdgeHeavy:
    def test_params_positive(self):
        p = compute_mpo_params_edge_heavy(64, 64, 8, 3)
        assert p > 0

    def test_params_less_than_dense(self):
        p = compute_mpo_params_edge_heavy(4096, 4096, 64, 3)
        assert p < 4096 * 4096
