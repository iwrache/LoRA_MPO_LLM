import pytest
import torch

from main_sa_htn_tucker_pilot import (
    apply_stage_permutation,
    build_stage_semantic_tensors,
    compute_stage_parameter_counts,
    parse_args,
    recover_stage_layer,
    resolve_pilot_layers,
)


def test_resolve_pilot_layers_defaults_to_middle_block():
    layers = resolve_pilot_layers(total_layers=32, start_layer=12, end_layer=15)
    assert layers == [12, 13, 14, 15]



def test_resolve_pilot_layers_rejects_invalid_range():
    with pytest.raises(ValueError):
        resolve_pilot_layers(total_layers=32, start_layer=15, end_layer=12)



def test_apply_stage_permutation_aligns_gate_up_rows_and_down_columns():
    gate = [torch.arange(12, dtype=torch.float32).reshape(4, 3)]
    up = [torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)]
    down = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)

    permuted = apply_stage_permutation(gate, up, down, perm)

    assert torch.equal(permuted["gate_perm"][0], gate[0][perm, :])
    assert torch.equal(permuted["up_perm"][0], up[0][perm, :])
    assert torch.equal(permuted["down_perm"][0], down[0][:, perm])



def test_build_stage_semantic_tensors_shapes():
    gate_perm = [torch.randn(11008, 4096) for _ in range(4)]
    up_perm = [torch.randn(11008, 4096) for _ in range(4)]
    down_perm = [torch.randn(4096, 11008) for _ in range(4)]

    upward, downward = build_stage_semantic_tensors(gate_perm, up_perm, down_perm)

    assert upward.shape == (4, 2, 11008, 4096)
    assert downward.shape == (4, 4096, 11008)



def test_recover_stage_layer_applies_inverse_permutation():
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    gate = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    up = torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)
    down = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    recovered = recover_stage_layer(gate, up, down, inv_perm)

    assert torch.equal(recovered["gate"], gate[inv_perm, :])
    assert torch.equal(recovered["up"], up[inv_perm, :])
    assert torch.equal(recovered["down"], down[:, inv_perm])



def test_compute_stage_parameter_counts_returns_positive_counts():
    counts = compute_stage_parameter_counts(
        num_stage_layers=4,
        d_out=11008,
        d_in=4096,
        rank_layer_up=2,
        rank_module_up=1,
        rank_layer_down=2,
        lora_rank=32,
    )

    assert counts["original_total"] > 0
    assert counts["compressed_total"] > 0
    assert counts["compression_ratio"] > 0



def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main_sa_htn_tucker_pilot.py"])
    args = parse_args()

    assert args.pilot_start_layer == 12
    assert args.pilot_end_layer == 15
    assert args.reduced_dim == 512
    assert args.rank_layer_up == 2
    assert args.rank_module_up == 1
    assert args.rank_layer_down == 2
    assert args.lora_rank == 32
    assert args.eval_after_tucker is True
