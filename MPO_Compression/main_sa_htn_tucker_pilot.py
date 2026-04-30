from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import leaves_list, linkage
from tensorly.decomposition import tucker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--pilot_start_layer", type=int, default=12)
    parser.add_argument("--pilot_end_layer", type=int, default=15)
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--calib_max_len", type=int, default=512)
    parser.add_argument("--reduced_dim", type=int, default=512)
    parser.add_argument("--perm_mode", type=str, default="stage")
    parser.add_argument("--disable_perm", action="store_true")
    parser.add_argument("--rank_layer_up", type=int, default=2)
    parser.add_argument("--rank_module_up", type=int, default=1)
    parser.add_argument("--rank_layer_down", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--local_steps", type=int, default=300)
    parser.add_argument("--distill_steps", type=int, default=1200)
    parser.add_argument("--eval_after_tucker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_max_tokens", type=int, default=15000)
    parser.add_argument("--eval_stride", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="./outputs/sa_htn_real_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_local_smoke", action="store_true")
    return parser.parse_args()


def resolve_pilot_layers(total_layers: int, start_layer: int, end_layer: int) -> list[int]:
    if start_layer < 0 or end_layer < 0 or start_layer > end_layer or end_layer >= total_layers:
        raise ValueError(f"Invalid pilot layer range: {start_layer}-{end_layer} for total_layers={total_layers}")
    return list(range(start_layer, end_layer + 1))


def load_models_and_tokenizer(model_name: str, device: torch.device) -> tuple[nn.Module, nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=None).to(device)
    student_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=None).to(device)
    teacher_model.eval()
    return teacher_model, student_model, tokenizer


def collect_activation_scales(
    student_model: nn.Module,
    tokenizer: Any,
    calib_samples: int,
    calib_max_len: int,
) -> dict[str, torch.Tensor]:
    from calibration import get_activation_scales

    return get_activation_scales(student_model, tokenizer, num_samples=calib_samples, max_len=calib_max_len)


def extract_stage_weights(student_model: nn.Module, layer_indices: list[int]) -> dict[str, list[torch.Tensor]]:
    gate_weights, up_weights, down_weights = [], [], []
    for layer_idx in layer_indices:
        layer = student_model.model.layers[layer_idx].mlp
        gate_weights.append(layer.gate_proj.weight.detach().clone())
        up_weights.append(layer.up_proj.weight.detach().clone())
        down_weights.append(layer.down_proj.weight.detach().clone())
    return {"gate": gate_weights, "up": up_weights, "down": down_weights}


def get_stage_permutation(
    stage_weights: dict[str, list[torch.Tensor]],
    activation_scales: dict[str, torch.Tensor],
    layer_indices: list[int],
    reduced_dim: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sketches = []
    for layer_idx, gate, up in zip(layer_indices, stage_weights["gate"], stage_weights["up"]):
        s_gate = activation_scales[f"model.layers.{layer_idx}.mlp.gate_proj"].to(gate.device).float()
        s_up = activation_scales[f"model.layers.{layer_idx}.mlp.up_proj"].to(up.device).float()
        features = torch.cat([gate.float() * s_gate.unsqueeze(0), up.float() * s_up.unsqueeze(0)], dim=1)
        projection = torch.randn(features.shape[1], reduced_dim, generator=generator, device=features.device, dtype=features.dtype)
        sketch = features @ projection
        print(
            f"layer={layer_idx} feature_shape={tuple(features.shape)} "
            f"projection_shape={tuple(projection.shape)} sketch_shape={tuple(sketch.shape)}"
        )
        sketches.append(sketch)
    stage_sketch = torch.stack(sketches, dim=0).mean(dim=0)
    linkage_matrix = linkage(stage_sketch.detach().cpu().numpy(), method="ward", metric="euclidean")
    perm_stage = torch.tensor(leaves_list(linkage_matrix), dtype=torch.long)
    inv_perm_stage = torch.argsort(perm_stage)
    return perm_stage, inv_perm_stage, stage_sketch


def apply_stage_permutation(
    gate_weights: list[torch.Tensor],
    up_weights: list[torch.Tensor],
    down_weights: list[torch.Tensor],
    perm_stage: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    return {
        "gate_perm": [weight[perm_stage, :] for weight in gate_weights],
        "up_perm": [weight[perm_stage, :] for weight in up_weights],
        "down_perm": [weight[:, perm_stage] for weight in down_weights],
    }


def build_stage_semantic_tensors(
    gate_perm: list[torch.Tensor],
    up_perm: list[torch.Tensor],
    down_perm: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    upward_tensor = torch.stack([torch.stack([gate, up], dim=0) for gate, up in zip(gate_perm, up_perm)], dim=0)
    downward_tensor = torch.stack(down_perm, dim=0)
    return upward_tensor, downward_tensor


def run_stage_tucker(
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


def recover_stage_layer(
    recon_gate: torch.Tensor,
    recon_up: torch.Tensor,
    recon_down: torch.Tensor,
    inv_perm_stage: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "gate": recon_gate[inv_perm_stage, :],
        "up": recon_up[inv_perm_stage, :],
        "down": recon_down[:, inv_perm_stage],
    }


def replace_stage_weights_in_student(
    student_model: nn.Module,
    layer_indices: list[int],
    recon_upward: torch.Tensor,
    recon_downward: torch.Tensor,
    inv_perm_stage: torch.Tensor,
) -> None:
    for offset, layer_idx in enumerate(layer_indices):
        recovered = recover_stage_layer(
            recon_upward[offset, 0],
            recon_upward[offset, 1],
            recon_downward[offset],
            inv_perm_stage,
        )
        layer = student_model.model.layers[layer_idx].mlp
        with torch.no_grad():
            layer.gate_proj.weight.copy_(recovered["gate"].to(layer.gate_proj.weight.dtype))
            layer.up_proj.weight.copy_(recovered["up"].to(layer.up_proj.weight.dtype))
            layer.down_proj.weight.copy_(recovered["down"].to(layer.down_proj.weight.dtype))


def compute_stage_parameter_counts(
    num_stage_layers: int,
    d_out: int,
    d_in: int,
    rank_layer_up: int,
    rank_module_up: int,
    rank_layer_down: int,
    lora_rank: int,
) -> dict[str, float]:
    original_up = num_stage_layers * 2 * d_out * d_in
    original_down = num_stage_layers * d_in * d_out
    original_total = original_up + original_down

    compressed_up = (
        rank_layer_up * rank_module_up * d_out * d_in
        + num_stage_layers * rank_layer_up
        + 2 * rank_module_up
        + d_out * d_out
        + d_in * d_in
    )
    compressed_down = (
        rank_layer_down * d_in * d_out
        + num_stage_layers * rank_layer_down
        + d_in * d_in
        + d_out * d_out
    )
    lora_params = num_stage_layers * 3 * lora_rank * (d_in + d_out)
    compressed_total = compressed_up + compressed_down + lora_params

    return {
        "original_up": original_up,
        "original_down": original_down,
        "original_total": original_total,
        "compressed_up": compressed_up,
        "compressed_down": compressed_down,
        "lora_params": lora_params,
        "compressed_total": compressed_total,
        "compression_ratio": original_total / compressed_total,
    }


def save_artifacts(
    save_dir: Path,
    metrics: dict[str, Any],
    perm_stage: torch.Tensor,
    inv_perm_stage: torch.Tensor,
    student_after_tucker: nn.Module | None,
    student_final: nn.Module | None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"perm_stage": perm_stage.cpu(), "inv_perm_stage": inv_perm_stage.cpu()}, save_dir / "perm_stage.pt")
    if student_after_tucker is not None:
        torch.save(student_after_tucker.state_dict(), save_dir / "student_after_tucker.pt")
    if student_final is not None:
        torch.save(student_final.state_dict(), save_dir / "student_final.pt")


class TuckerLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, lora_rank: int, s_vector: torch.Tensor | None = None):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        out_features, in_features = self.base.weight.shape
        dtype = self.base.weight.dtype
        device = self.base.weight.device

        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=device, dtype=dtype))

        with torch.no_grad():
            delta = self.base.weight.detach().float()
            if s_vector is not None:
                s = s_vector.to(device).float()
                delta = delta * s.unsqueeze(0)
            u, svals, vh = torch.linalg.svd(delta, full_matrices=False)
            s_sqrt = torch.diag(torch.sqrt(svals[:lora_rank].clamp(min=0)))
            a_scaled = s_sqrt @ vh[:lora_rank, :]
            b_mat = u[:, :lora_rank] @ s_sqrt
            if s_vector is not None:
                a_scaled = a_scaled / s.unsqueeze(0)
            self.lora_A.copy_(a_scaled.to(dtype))
            self.lora_B.copy_(b_mat.to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return base_out + lora_out


def main() -> None:
    args = parse_args()
    if args.dry_local_smoke:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        gate_perm = [torch.randn(11008, 4096) for _ in range(4)]
        up_perm = [torch.randn(11008, 4096) for _ in range(4)]
        down_perm = [torch.randn(4096, 11008) for _ in range(4)]
        upward_tensor, downward_tensor = build_stage_semantic_tensors(gate_perm, up_perm, down_perm)
        result = run_stage_tucker(upward_tensor[:, :, :32, :16], downward_tensor[:, :16, :32], 2, 1, 2)
        print(tuple(result["upward_recon"].shape))
        print(tuple(result["downward_recon"].shape))
        return

    from accelerate import Accelerator
    from healing import train_healing
    from main_progressive_with_e2e import eval_ppl

    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")
    device = accelerator.device
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    teacher_model, student_model, tokenizer = load_models_and_tokenizer(args.model_name, device)
    layer_indices = resolve_pilot_layers(len(student_model.model.layers), args.pilot_start_layer, args.pilot_end_layer)

    print("=== SA-HTN Real 4-Layer Pilot ===")
    print(f"Pilot layers: {layer_indices}")
    print(f"Model: {args.model_name}")
    print(f"Save dir: {args.save_dir}")

    activation_scales = collect_activation_scales(student_model, tokenizer, args.calib_samples, args.calib_max_len)
    stage_weights = extract_stage_weights(student_model, layer_indices)

    if args.disable_perm:
        hidden_dim = stage_weights["gate"][0].shape[0]
        perm_stage = torch.arange(hidden_dim, dtype=torch.long)
        inv_perm_stage = perm_stage.clone()
    else:
        perm_stage, inv_perm_stage, _ = get_stage_permutation(
            stage_weights,
            activation_scales,
            layer_indices,
            args.reduced_dim,
            generator,
        )

    permuted = apply_stage_permutation(stage_weights["gate"], stage_weights["up"], stage_weights["down"], perm_stage)
    upward_tensor, downward_tensor = build_stage_semantic_tensors(permuted["gate_perm"], permuted["up_perm"], permuted["down_perm"])
    tucker_result = run_stage_tucker(upward_tensor, downward_tensor, args.rank_layer_up, args.rank_module_up, args.rank_layer_down)
    replace_stage_weights_in_student(student_model, layer_indices, tucker_result["upward_recon"], tucker_result["downward_recon"], inv_perm_stage)

    ppl_after_tucker = None
    if args.eval_after_tucker:
        ppl_after_tucker = eval_ppl(student_model, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        print(f"[Checkpoint] Tucker-only PPL: {ppl_after_tucker}")

    student_after_tucker = student_model

    for layer_idx in layer_indices:
        mlp = student_model.model.layers[layer_idx].mlp
        gate_scale = activation_scales.get(f"model.layers.{layer_idx}.mlp.gate_proj")
        up_scale = activation_scales.get(f"model.layers.{layer_idx}.mlp.up_proj")
        down_scale = activation_scales.get(f"model.layers.{layer_idx}.mlp.down_proj")
        mlp.gate_proj = TuckerLoRALinear(mlp.gate_proj, args.lora_rank, gate_scale)
        mlp.up_proj = TuckerLoRALinear(mlp.up_proj, args.lora_rank, up_scale)
        mlp.down_proj = TuckerLoRALinear(mlp.down_proj, args.lora_rank, down_scale)

    student_model = train_healing(
        student_model=student_model,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        max_update_steps=args.distill_steps,
        accelerator=accelerator,
        tune_mpo=False,
    )

    ppl_final = eval_ppl(student_model, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
    print(f"[Checkpoint] Final PPL: {ppl_final}")

    counts = compute_stage_parameter_counts(
        num_stage_layers=len(layer_indices),
        d_out=stage_weights["gate"][0].shape[0],
        d_in=stage_weights["gate"][0].shape[1],
        rank_layer_up=args.rank_layer_up,
        rank_module_up=args.rank_module_up,
        rank_layer_down=args.rank_layer_down,
        lora_rank=args.lora_rank,
    )
    metrics = {
        "model_name": args.model_name,
        "pilot_layers": layer_indices,
        "reduced_dim": args.reduced_dim,
        "perm_enabled": not args.disable_perm,
        "rank_layer_up": args.rank_layer_up,
        "rank_module_up": args.rank_module_up,
        "rank_layer_down": args.rank_layer_down,
        "lora_rank": args.lora_rank,
        "ppl_after_tucker": ppl_after_tucker,
        "ppl_final": ppl_final,
        **counts,
    }
    save_artifacts(Path(args.save_dir), metrics, perm_stage, inv_perm_stage, student_after_tucker, student_model)


if __name__ == "__main__":
    main()
