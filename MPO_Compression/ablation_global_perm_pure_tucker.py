from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import tensorly as tl
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from tensorly.decomposition import tucker


DEFAULT_MAX_ALLOC_GB = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--d-out", type=int, default=11008)
    parser.add_argument("--reduced-dim", type=int, default=512)
    parser.add_argument("--rank-layer-up", type=int, default=2)
    parser.add_argument("--rank-module-up", type=int, default=1)
    parser.add_argument("--rank-layer-down", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-alloc-gb", type=float, default=DEFAULT_MAX_ALLOC_GB)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def estimate_tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dtype).element_size() * np.prod(shape))


def ensure_memory_budget(num_layers: int, d_out: int, d_in: int, dtype: torch.dtype, max_alloc_gb: float) -> None:
    upward_shape = (num_layers, 2, d_out, d_in)
    downward_shape = (num_layers, d_in, d_out)
    upward_bytes = estimate_tensor_bytes(upward_shape, dtype)
    downward_bytes = estimate_tensor_bytes(downward_shape, dtype)
    total_gb = (upward_bytes + downward_bytes) / (1024**3)
    print(f"estimated_upward_tensor_gb={upward_bytes / (1024**3):.3f}")
    print(f"estimated_downward_tensor_gb={downward_bytes / (1024**3):.3f}")
    print(f"estimated_total_tensor_gb={total_gb:.3f}")
    if total_gb > max_alloc_gb:
        raise MemoryError(
            "Requested tensorization is likely to OOM on this machine. "
            f"Estimated semantic tensor footprint is {total_gb:.3f} GiB, which exceeds --max-alloc-gb={max_alloc_gb}. "
            "Lower --num-layers/--d-in/--d-out, switch dtype, or run on a larger-memory server."
        )


def generate_mock_weights(
    num_layers: int,
    d_out: int,
    d_in: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, list[torch.Tensor]]:
    return {
        "W_gate": [torch.randn(d_out, d_in, device=device, dtype=dtype) for _ in range(num_layers)],
        "W_up": [torch.randn(d_out, d_in, device=device, dtype=dtype) for _ in range(num_layers)],
        "W_down": [torch.randn(d_in, d_out, device=device, dtype=dtype) for _ in range(num_layers)],
        "s_vec_gate": [torch.randn(d_in, device=device, dtype=dtype) for _ in range(num_layers)],
        "s_vec_up": [torch.randn(d_in, device=device, dtype=dtype) for _ in range(num_layers)],
    }


def get_global_permutation(
    W_gate: list[torch.Tensor],
    W_up: list[torch.Tensor],
    s_vec_gate: list[torch.Tensor],
    s_vec_up: list[torch.Tensor],
    reduced_dim: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sketches: list[torch.Tensor] = []
    for layer_idx, (gate, up, s_gate, s_up) in enumerate(zip(W_gate, W_up, s_vec_gate, s_vec_up)):
        features = torch.cat([gate * s_gate.unsqueeze(0), up * s_up.unsqueeze(0)], dim=1)
        projection = torch.randn(
            features.shape[1],
            reduced_dim,
            generator=generator,
            device=features.device,
            dtype=features.dtype,
        )
        sketch = features @ projection
        print(
            f"layer={layer_idx} feature_shape={tuple(features.shape)} "
            f"projection_shape={tuple(projection.shape)} sketch_shape={tuple(sketch.shape)}"
        )
        sketches.append(sketch)
    global_sketch = torch.stack(sketches, dim=0).mean(dim=0)
    linkage_matrix = linkage(global_sketch.detach().cpu().numpy(), method="ward", metric="euclidean")
    perm_global = torch.tensor(leaves_list(linkage_matrix), dtype=torch.long)
    inv_perm_global = torch.argsort(perm_global)
    return perm_global, inv_perm_global, global_sketch


def apply_global_permutation(
    W_gate: list[torch.Tensor],
    W_up: list[torch.Tensor],
    W_down: list[torch.Tensor],
    perm_global: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    return {
        "W_gate_perm": [weight[perm_global, :] for weight in W_gate],
        "W_up_perm": [weight[perm_global, :] for weight in W_up],
        "W_down_perm": [weight[:, perm_global] for weight in W_down],
    }


def build_semantic_tensors(
    W_gate_perm: list[torch.Tensor],
    W_up_perm: list[torch.Tensor],
    W_down_perm: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    upward_tensor = torch.stack(
        [torch.stack([gate, up], dim=0) for gate, up in zip(W_gate_perm, W_up_perm)],
        dim=0,
    )
    downward_tensor = torch.stack(W_down_perm, dim=0)
    return upward_tensor, downward_tensor


def run_tucker_decomposition(
    upward_tensor: torch.Tensor,
    downward_tensor: torch.Tensor,
    rank_layer_up: int,
    rank_module_up: int,
    rank_layer_down: int,
) -> dict[str, Any]:
    tl.set_backend("pytorch")

    upward_ranks = [rank_layer_up, rank_module_up, upward_tensor.shape[2], upward_tensor.shape[3]]
    downward_ranks = [rank_layer_down, downward_tensor.shape[1], downward_tensor.shape[2]]

    upward_core, upward_factors = tucker(upward_tensor, rank=upward_ranks, init="svd")
    downward_core, downward_factors = tucker(downward_tensor, rank=downward_ranks, init="svd")

    upward_recon = tl.tucker_to_tensor((upward_core, upward_factors))
    downward_recon = tl.tucker_to_tensor((downward_core, downward_factors))

    return {
        "upward_core": upward_core,
        "upward_factors": upward_factors,
        "upward_recon": upward_recon,
        "downward_core": downward_core,
        "downward_factors": downward_factors,
        "downward_recon": downward_recon,
    }


def recover_layer_zero(
    recon_gate_0: torch.Tensor,
    recon_up_0: torch.Tensor,
    recon_down_0: torch.Tensor,
    inv_perm_global: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "gate_0_recovered": recon_gate_0[inv_perm_global, :],
        "up_0_recovered": recon_up_0[inv_perm_global, :],
        "down_0_recovered": recon_down_0[:, inv_perm_global],
    }


def compute_parameter_counts(
    num_layers: int,
    d_out: int,
    d_in: int,
    rank_layer_up: int,
    rank_module_up: int,
    rank_layer_down: int,
) -> dict[str, float]:
    original_up = num_layers * 2 * d_out * d_in
    original_down = num_layers * d_in * d_out
    original_total = original_up + original_down

    compressed_up = (
        rank_layer_up * rank_module_up * d_out * d_in
        + num_layers * rank_layer_up
        + 2 * rank_module_up
        + d_out * d_out
        + d_in * d_in
    )
    compressed_down = (
        rank_layer_down * d_in * d_out
        + num_layers * rank_layer_down
        + d_in * d_in
        + d_out * d_out
    )
    compressed_total = compressed_up + compressed_down

    return {
        "original_up": original_up,
        "original_down": original_down,
        "original_total": original_total,
        "compressed_up": compressed_up,
        "compressed_down": compressed_down,
        "compressed_total": compressed_total,
        "compression_ratio": original_total / compressed_total,
    }


def compute_reconstruction_metrics(
    W_gate_0: torch.Tensor,
    W_up_0: torch.Tensor,
    W_down_0: torch.Tensor,
    gate_0_recovered: torch.Tensor,
    up_0_recovered: torch.Tensor,
    down_0_recovered: torch.Tensor,
) -> dict[str, float]:
    mse_gate_0 = torch.mean((W_gate_0 - gate_0_recovered) ** 2).item()
    mse_up_0 = torch.mean((W_up_0 - up_0_recovered) ** 2).item()
    mse_down_0 = torch.mean((W_down_0 - down_0_recovered) ** 2).item()
    mean_layer0_mse = (mse_gate_0 + mse_up_0 + mse_down_0) / 3.0
    return {
        "mse_gate_0": mse_gate_0,
        "mse_up_0": mse_up_0,
        "mse_down_0": mse_down_0,
        "mean_layer0_mse": mean_layer0_mse,
    }


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== SA-HTN Ablation #1 ===")
    print(f"num_layers={args.num_layers}, d_in={args.d_in}, d_out={args.d_out}, reduced_dim={args.reduced_dim}")
    print(
        f"rank_layer_up={args.rank_layer_up}, rank_module_up={args.rank_module_up}, "
        f"rank_layer_down={args.rank_layer_down}"
    )
    print(f"device={args.device}, dtype={dtype}, seed={args.seed}, max_alloc_gb={args.max_alloc_gb}")

    ensure_memory_budget(args.num_layers, args.d_out, args.d_in, dtype, args.max_alloc_gb)

    weights = generate_mock_weights(args.num_layers, args.d_out, args.d_in, args.device, dtype)
    print(f"W_gate[0].shape={tuple(weights['W_gate'][0].shape)}")
    print(f"W_up[0].shape={tuple(weights['W_up'][0].shape)}")
    print(f"W_down[0].shape={tuple(weights['W_down'][0].shape)}")
    print(f"s_vec_gate[0].shape={tuple(weights['s_vec_gate'][0].shape)}")
    print(f"s_vec_up[0].shape={tuple(weights['s_vec_up'][0].shape)}")

    perm_global, inv_perm_global, global_sketch = get_global_permutation(
        weights["W_gate"],
        weights["W_up"],
        weights["s_vec_gate"],
        weights["s_vec_up"],
        args.reduced_dim,
        generator,
    )
    print(f"global_sketch.shape={tuple(global_sketch.shape)}")
    print(f"perm_global.shape={tuple(perm_global.shape)}")
    print(f"inv_perm_global.shape={tuple(inv_perm_global.shape)}")

    permuted = apply_global_permutation(weights["W_gate"], weights["W_up"], weights["W_down"], perm_global)
    upward_tensor, downward_tensor = build_semantic_tensors(
        permuted["W_gate_perm"],
        permuted["W_up_perm"],
        permuted["W_down_perm"],
    )
    print(f"upward_tensor.shape={tuple(upward_tensor.shape)}")
    print(f"downward_tensor.shape={tuple(downward_tensor.shape)}")

    decomp = run_tucker_decomposition(
        upward_tensor,
        downward_tensor,
        args.rank_layer_up,
        args.rank_module_up,
        args.rank_layer_down,
    )
    print(f"upward_core.shape={tuple(decomp['upward_core'].shape)}")
    print(f"downward_core.shape={tuple(decomp['downward_core'].shape)}")
    print(f"upward_recon.shape={tuple(decomp['upward_recon'].shape)}")
    print(f"downward_recon.shape={tuple(decomp['downward_recon'].shape)}")

    recovered = recover_layer_zero(
        decomp["upward_recon"][0, 0],
        decomp["upward_recon"][0, 1],
        decomp["downward_recon"][0],
        inv_perm_global,
    )
    counts = compute_parameter_counts(
        args.num_layers,
        args.d_out,
        args.d_in,
        args.rank_layer_up,
        args.rank_module_up,
        args.rank_layer_down,
    )
    metrics = compute_reconstruction_metrics(
        weights["W_gate"][0],
        weights["W_up"][0],
        weights["W_down"][0],
        recovered["gate_0_recovered"],
        recovered["up_0_recovered"],
        recovered["down_0_recovered"],
    )

    print(f"Original Parameter Count: {counts['original_total']}")
    print(f"Compressed Parameter Count: {counts['compressed_total']}")
    print(f"Compression Ratio: {counts['compression_ratio']:.6f}")
    print(f"MSE(gate_0): {metrics['mse_gate_0']:.6f}")
    print(f"MSE(up_0): {metrics['mse_up_0']:.6f}")
    print(f"MSE(down_0): {metrics['mse_down_0']:.6f}")
    print(f"Mean Layer-0 MSE: {metrics['mean_layer0_mse']:.6f}")


if __name__ == "__main__":
    main()
