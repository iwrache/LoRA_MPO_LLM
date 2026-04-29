import pytest
import torch

from ablation_global_perm_pure_tucker import (
    apply_global_permutation,
    build_semantic_tensors,
    compute_parameter_counts,
    ensure_memory_budget,
    generate_mock_weights,
    parse_args,
    recover_layer_zero,
    run_tucker_decomposition,
)


def test_generate_mock_weights_shapes():
    weights = generate_mock_weights(num_layers=4, d_out=11008, d_in=4096, device="cpu", dtype=torch.float32)

    assert len(weights["W_gate"]) == 4
    assert len(weights["W_up"]) == 4
    assert len(weights["W_down"]) == 4
    assert len(weights["s_vec_gate"]) == 4
    assert len(weights["s_vec_up"]) == 4

    assert weights["W_gate"][0].shape == (11008, 4096)
    assert weights["W_up"][0].shape == (11008, 4096)
    assert weights["W_down"][0].shape == (4096, 11008)
    assert weights["s_vec_gate"][0].shape == (4096,)
    assert weights["s_vec_up"][0].shape == (4096,)



def test_apply_global_permutation_aligns_upward_rows_and_downward_columns():
    gate = [torch.arange(12, dtype=torch.float32).reshape(4, 3)]
    up = [torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)]
    down = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)

    permuted = apply_global_permutation(gate, up, down, perm)

    assert torch.equal(permuted["W_gate_perm"][0], gate[0][perm, :])
    assert torch.equal(permuted["W_up_perm"][0], up[0][perm, :])
    assert torch.equal(permuted["W_down_perm"][0], down[0][:, perm])



def test_build_semantic_tensors_shapes():
    gate_perm = [torch.randn(11008, 4096) for _ in range(4)]
    up_perm = [torch.randn(11008, 4096) for _ in range(4)]
    down_perm = [torch.randn(4096, 11008) for _ in range(4)]

    upward, downward = build_semantic_tensors(gate_perm, up_perm, down_perm)

    assert upward.shape == (4, 2, 11008, 4096)
    assert downward.shape == (4, 4096, 11008)



def test_recover_layer_zero_applies_inverse_permutation_correctly():
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv_perm = torch.argsort(perm)

    recon_gate_0 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    recon_up_0 = torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)
    recon_down_0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    recovered = recover_layer_zero(recon_gate_0, recon_up_0, recon_down_0, inv_perm)

    assert torch.equal(recovered["gate_0_recovered"], recon_gate_0[inv_perm, :])
    assert torch.equal(recovered["up_0_recovered"], recon_up_0[inv_perm, :])
    assert torch.equal(recovered["down_0_recovered"], recon_down_0[:, inv_perm])



def test_compute_parameter_counts_matches_expected_formula():
    counts = compute_parameter_counts(
        num_layers=4,
        d_out=11008,
        d_in=4096,
        rank_layer_up=2,
        rank_module_up=1,
        rank_layer_down=2,
    )

    expected_original_up = 4 * 2 * 11008 * 4096
    expected_original_down = 4 * 4096 * 11008
    expected_original_total = expected_original_up + expected_original_down

    expected_compressed_up = (
        2 * 1 * 11008 * 4096
        + 4 * 2
        + 2 * 1
        + 11008 * 11008
        + 4096 * 4096
    )
    expected_compressed_down = (
        2 * 4096 * 11008
        + 4 * 2
        + 4096 * 4096
        + 11008 * 11008
    )
    expected_compressed_total = expected_compressed_up + expected_compressed_down

    assert counts["original_up"] == expected_original_up
    assert counts["original_down"] == expected_original_down
    assert counts["original_total"] == expected_original_total
    assert counts["compressed_up"] == expected_compressed_up
    assert counts["compressed_down"] == expected_compressed_down
    assert counts["compressed_total"] == expected_compressed_total
    assert counts["compression_ratio"] == expected_original_total / expected_compressed_total



def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ablation_global_perm_pure_tucker.py"])
    args = parse_args()

    assert args.num_layers == 4
    assert args.d_in == 4096
    assert args.d_out == 11008
    assert args.reduced_dim == 512
    assert args.rank_layer_up == 2
    assert args.rank_module_up == 1
    assert args.rank_layer_down == 2
    assert args.device == "cpu"
    assert args.dtype == "float32"



def test_run_tucker_decomposition_reconstructs_small_tensors():
    upward = torch.randn(2, 2, 8, 4)
    downward = torch.randn(2, 4, 8)

    result = run_tucker_decomposition(
        upward_tensor=upward,
        downward_tensor=downward,
        rank_layer_up=1,
        rank_module_up=1,
        rank_layer_down=1,
    )

    assert result["upward_recon"].shape == upward.shape
    assert result["downward_recon"].shape == downward.shape
    assert result["upward_core"].shape == (1, 1, 8, 4)
    assert result["downward_core"].shape == (1, 4, 8)



def test_ensure_memory_budget_raises_for_large_default_shapes():
    with pytest.raises(MemoryError):
        ensure_memory_budget(num_layers=4, d_out=11008, d_in=4096, dtype=torch.float32, max_alloc_gb=1.0)
