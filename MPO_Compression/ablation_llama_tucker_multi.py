#!/usr/bin/env python3
"""
SA-HTN Multi-Stage Pilot (ASVD + Permutation + Tucker + Healing)
支持同时压缩多个 layer stage，每个 stage 独立排列与 Tucker 分解。
恢复阶段在所有压缩层上统一进行。
增加: 4个阶段的完整PPL评测与全模型压缩率统计。
"""

from __future__ import annotations

import os
os.environ["HF_HOME"] = "/mnt/hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hf-cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from scipy.cluster.hierarchy import leaves_list, linkage
from tensorly.decomposition import tucker
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 复用已有的基础设施
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
from calibration import get_activation_scales
from healing import train_healing


# ---------- 命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    # 支持多个 stage: 例如 --stage_ranges 10-14 15-19
    parser.add_argument("--stage_ranges", type=str, nargs='+', default=["12-15"],
                        help="空格分隔的区间列表，例如 '10-14' '15-19'")
    parser.add_argument("--save_dir", type=str, default="./outputs/sa_htn_real_pilot")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--calib_max_len", type=int, default=512)
    parser.add_argument("--reduced_dim", type=int, default=512)
    parser.add_argument("--disable_perm", action="store_true")

    parser.add_argument("--rank_layer_up", type=int, default=2)
    parser.add_argument("--rank_module_up", type=int, default=1)
    parser.add_argument("--rank_layer_down", type=int, default=2)
    parser.add_argument("--rank_out_up", type=int, default=512)
    parser.add_argument("--rank_in_up", type=int, default=256)
    parser.add_argument("--rank_in_down", type=int, default=256)
    parser.add_argument("--rank_out_down", type=int, default=512)

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--layerwise_steps", type=int, default=100)
    parser.add_argument("--distill_steps", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--save_every_n_steps", type=int, default=500)

    # 4 个评测阶段的开关
    parser.add_argument("--eval_original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_after_tucker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_post_layerwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_max_tokens", type=int, default=15000)
    parser.add_argument("--eval_stride", type=int, default=512)
    parser.add_argument("--dry_local_smoke", action="store_true")
    return parser.parse_args()


# ---------- 解析 stage ranges ----------
def parse_stage_ranges(stage_ranges_str: list[str]) -> list[tuple[int, int]]:
    ranges = []
    for s in stage_ranges_str:
        start, end = s.split('-')
        ranges.append((int(start), int(end)))
    return ranges


# ---------- 模型加载 ----------
def load_models_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{{ eos_token }}\n"
            "{% endif %}"
            "{% endfor %}"
        )

    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=None).to(device)
    student = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=None).to(device)
    teacher.eval()
    return teacher, student, tokenizer


# ---------- 权重提取与 ASVD 处理 ----------
def extract_stage_weights(model, layer_indices):
    gate_w, up_w, down_w = [], [], []
    for idx in layer_indices:
        mlp = model.model.layers[idx].mlp
        gate_w.append(mlp.gate_proj.weight.detach().clone())
        up_w.append(mlp.up_proj.weight.detach().clone())
        down_w.append(mlp.down_proj.weight.detach().clone())
    return {"gate": gate_w, "up": up_w, "down": down_w}


def apply_asvd_scaling(stage_weights, scales, layer_indices):
    scaled = {"gate": [], "up": [], "down": []}
    for offset, idx in enumerate(layer_indices):
        gate, up, down = stage_weights["gate"][offset], stage_weights["up"][offset], stage_weights["down"][offset]
        s_gate = scales[f"model.layers.{idx}.mlp.gate_proj"].to(gate.device).float()
        s_up = scales[f"model.layers.{idx}.mlp.up_proj"].to(up.device).float()
        s_down = scales[f"model.layers.{idx}.mlp.down_proj"].to(down.device).float()
        scaled["gate"].append(gate.float() * s_gate.unsqueeze(0))
        scaled["up"].append(up.float() * s_up.unsqueeze(0))
        scaled["down"].append(down.float() * s_down.unsqueeze(0))
    return scaled


# ---------- 排列 ----------
def get_stage_permutation(scaled_weights, layer_indices, reduced_dim, generator):
    sketches = []
    for gate_s, up_s in zip(scaled_weights["gate"], scaled_weights["up"]):
        features = torch.cat([gate_s, up_s], dim=1)
        proj = torch.randn(features.shape[1], reduced_dim, generator=generator, device=features.device, dtype=features.dtype)
        proj /= math.sqrt(reduced_dim)
        sketches.append(features @ proj)
    stage_sketch = torch.stack(sketches, dim=0).mean(dim=0)
    linkage_matrix = linkage(stage_sketch.detach().cpu().numpy(), method="ward", metric="euclidean")
    perm = torch.tensor(leaves_list(linkage_matrix), dtype=torch.long)
    inv_perm = torch.argsort(perm)
    return perm, inv_perm


def apply_stage_permutation(scaled_weights, perm):
    return {
        "gate_perm": [w[perm, :] for w in scaled_weights["gate"]],
        "up_perm": [w[perm, :] for w in scaled_weights["up"]],
        "down_perm": [w[:, perm] for w in scaled_weights["down"]],
    }


# ---------- Tucker 分解 ----------
def run_stage_tucker(upward_tensor, downward_tensor, args):
    tl.set_backend("pytorch")
    upward_ranks = [args.rank_layer_up, args.rank_module_up, args.rank_out_up, args.rank_in_up]
    downward_ranks = [args.rank_layer_down, args.rank_in_down, args.rank_out_down]
    print(f"  -> Tucker Up Ranks: {upward_ranks}, Down Ranks: {downward_ranks}")
    up_core, up_factors = tucker(upward_tensor.float(), rank=upward_ranks, init="svd")
    down_core, down_factors = tucker(downward_tensor.float(), rank=downward_ranks, init="svd")
    return {
        "upward_recon": tl.tucker_to_tensor((up_core, up_factors)),
        "downward_recon": tl.tucker_to_tensor((down_core, down_factors)),
    }


# ---------- 逆 ASVD 与权重替换 ----------
def replace_stage_weights_in_student(model, layer_indices, recon_up, recon_down, inv_perm, scales):
    for offset, idx in enumerate(layer_indices):
        gate_recon = recon_up[offset, 0][inv_perm, :]
        up_recon = recon_up[offset, 1][inv_perm, :]
        down_recon = recon_down[offset][:, inv_perm]
        mlp = model.model.layers[idx].mlp
        s_gate = scales[f"model.layers.{idx}.mlp.gate_proj"].to(gate_recon.device).float()
        s_up = scales[f"model.layers.{idx}.mlp.up_proj"].to(up_recon.device).float()
        s_down = scales[f"model.layers.{idx}.mlp.down_proj"].to(down_recon.device).float()
        with torch.no_grad():
            mlp.gate_proj.weight.copy_((gate_recon / s_gate.unsqueeze(0)).to(mlp.gate_proj.weight.dtype))
            mlp.up_proj.weight.copy_((up_recon / s_up.unsqueeze(0)).to(mlp.up_proj.weight.dtype))
            mlp.down_proj.weight.copy_((down_recon / s_down.unsqueeze(0)).to(mlp.down_proj.weight.dtype))


# ---------- LoRA 模块 ----------
class TuckerLoRALinear(nn.Module):
    def __init__(self, base_linear, lora_rank):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        out_f, in_f = self.base.weight.shape
        dtype, device = self.base.weight.dtype, self.base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, lora_rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))


# ---------- PPL 评测 ----------
@torch.no_grad()
def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=15000):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    device = model.model.embed_tokens.weight.device
    input_ids = enc.input_ids.to(device)
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    effective_max = min(max_len, max_ctx)
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]
    total_loss = 0.0
    total_tokens = 0
    for begin in tqdm(range(0, seq_len, stride), desc="PPL", leave=False):
        end = min(begin + effective_max, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1:
            continue
        out = model(chunk)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = chunk[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    if total_tokens == 0:
        return float("nan")
    return round(torch.exp(torch.tensor(total_loss / total_tokens)).item(), 2)


# ---------- 构建逐层恢复校准输入 ----------
def build_layerwise_calib_inputs(tokenizer, calib_samples: int, calib_max_len: int):
    calib_texts = []
    wiki_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_target = max(1, calib_samples // 2)
    for text in wiki_ds["text"]:
        if len(text) > 200:
            calib_texts.append(text)
        if len(calib_texts) >= wiki_target:
            break
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    for item in chat_ds:
        text = "\n".join([m["content"] for m in item["messages"]])
        if len(text) > 200:
            calib_texts.append(text)
        if len(calib_texts) >= calib_samples:
            break
    return [
        tokenizer(text, return_tensors="pt", max_length=calib_max_len, truncation=True).input_ids
        for text in calib_texts
    ]


# ---------- 逐层恢复 ----------
def run_layerwise_healing(student, teacher, calib_inputs, layer_indices, steps_per_layer, lr, accelerator, sync_dir: Path):
    print("\n[Phase 9] Strict Layerwise Recovery (Sequential)...")
    student.train()
    teacher.eval()

    class StopForwardException(Exception):
        pass

    for target_idx in layer_indices:
        if accelerator.is_main_process:
            print(f"\n  >>> 正在独立修复 Layer {target_idx} <<<")

        for p in student.parameters():
            p.requires_grad = False

        lora_params = []
        for n, p in student.named_parameters():
            if f"layers.{target_idx}." in n and "lora_" in n:
                p.requires_grad = True
                lora_params.append(p)

        optimizer = torch.optim.AdamW(lora_params, lr=lr)
        student_layer = student.model.layers[target_idx]
        teacher_layer = teacher.model.layers[target_idx]
        layer_device = next(student_layer.parameters()).device
        teacher_device = next(teacher_layer.parameters()).device

        cached_mlp_inputs = []
        def capture_hook(module, args):
            flattened_x = args[0].detach().cpu().reshape(-1, args[0].shape[-1])
            cached_mlp_inputs.append(flattened_x)
            raise StopForwardException

        handle = student_layer.mlp.register_forward_pre_hook(capture_hook)
        embed_device = student.model.embed_tokens.weight.device

        with torch.no_grad():
            for input_ids in tqdm(calib_inputs, desc=f"积累 Layer {target_idx} 输入", leave=False, disable=not accelerator.is_main_process):
                try:
                    student(input_ids.to(embed_device))
                except StopForwardException:
                    pass
        handle.remove()

        x_calib_tensor = torch.cat(cached_mlp_inputs, dim=0)
        num_tokens = x_calib_tensor.shape[0]

        student_layer.mlp.train()
        for step in range(steps_per_layer):
            optimizer.zero_grad()
            torch.manual_seed(42 + target_idx * 10000 + step)
            sample_indices = torch.randperm(num_tokens)[: min(2048, num_tokens)]
            current_dtype = next(student_layer.parameters()).dtype
            x_batch = x_calib_tensor[sample_indices].to(device=layer_device, dtype=current_dtype)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    y_orig = teacher_layer.mlp(x_batch.to(teacher_device)).to(layer_device)
                y_recon = student_layer.mlp(x_batch)
                loss = F.mse_loss(y_recon, y_orig)

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            if accelerator.is_main_process and step % 10 == 0:
                print(f"      Step [{step}/{steps_per_layer}] | Layer {target_idx} MSE: {loss.item():.5f}")

        student_layer.mlp.eval()
        del optimizer, cached_mlp_inputs, x_calib_tensor
        torch.cuda.empty_cache()
        gc.collect()

        # 层间同步
        sync_file = sync_dir / "tmp_lora_sync_tucker.pt"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            lora_state = {}
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wrapper = getattr(student.model.layers[target_idx].mlp, proj)
                prefix = f"model.layers.{target_idx}.mlp.{proj}."
                for key, value in wrapper.state_dict().items():
                    if "lora" in key:
                        lora_state[prefix + key] = value.cpu().clone()
            torch.save(lora_state, sync_file)
        accelerator.wait_for_everyone()
        if sync_file.exists():
            lora_dict = torch.load(sync_file, map_location=accelerator.device)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wrapper = getattr(student.model.layers[target_idx].mlp, proj)
                prefix = f"model.layers.{target_idx}.mlp.{proj}."
                state = {k[len(prefix):]: v for k, v in lora_dict.items() if k.startswith(prefix)}
                if state:
                    wrapper.load_state_dict(state, strict=False)
            del lora_dict
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"      [Sync] Layer {target_idx} parameters synced across GPUs.")
    print("[Phase 9] Layerwise Recovery done.\n")


# ---------- 指标与产物 ----------
def compute_stage_parameter_counts(
    stage_ranges: list[tuple[int, int]],
    d_out: int,
    d_in: int,
    rank_layer_up: int,
    rank_module_up: int,
    rank_out_up: int,
    rank_in_up: int,
    rank_layer_down: int,
    rank_in_down: int,
    rank_out_down: int,
    lora_rank: int,
) -> dict[str, float]:
    
    original_up = 0
    original_down = 0
    compressed_up = 0
    compressed_down = 0
    lora_params = 0

    for s_start, s_end in stage_ranges:
        num_stage_layers = s_end - s_start + 1
        original_up += num_stage_layers * 2 * d_out * d_in
        original_down += num_stage_layers * d_in * d_out

        compressed_up += (
            (rank_layer_up * rank_module_up * rank_out_up * rank_in_up) # Core
            + (num_stage_layers * rank_layer_up)                        # Layer Factor
            + (2 * rank_module_up)                                      # Module Factor
            + (d_out * rank_out_up)                                     # Out Neuron Factor
            + (d_in * rank_in_up)                                       # In Neuron Factor
        )
        
        compressed_down += (
            (rank_layer_down * rank_in_down * rank_out_down)            # Core
            + (num_stage_layers * rank_layer_down)                      # Layer Factor
            + (d_in * rank_in_down)                                     # In Neuron Factor
            + (d_out * rank_out_down)                                   # Out Neuron Factor
        )
        
        lora_params += num_stage_layers * 3 * lora_rank * (d_in + d_out)

    original_total = original_up + original_down
    compressed_total = compressed_up + compressed_down + lora_params

    return {
        "original_total": original_total,
        "compressed_up": compressed_up,
        "compressed_down": compressed_down,
        "lora_params": lora_params,
        "compressed_total": compressed_total,
        "compression_ratio": original_total / compressed_total,
    }


def save_artifacts(save_dir: Path, metrics: dict, perm_stages: dict | None, student_after_tucker_state: dict | None, student_final_model) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if perm_stages is not None:
        torch.save(perm_stages, save_dir / "perm_stages.pt")
    if student_after_tucker_state is not None:
        torch.save(student_after_tucker_state, save_dir / "student_after_tucker.pt")
    if student_final_model is not None:
        torch.save(student_final_model.state_dict(), save_dir / "student_final.pt")


# ---------- 主流程 ----------
def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")
    device = accelerator.device
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    teacher, student, tokenizer = load_models_and_tokenizer(args.model_name, device)
    total_layers = len(student.model.layers)

    # 在执行任何修改前，获取原模型的总体参数量
    total_params_orig = sum(p.numel() for p in student.parameters())

    stage_ranges = parse_stage_ranges(args.stage_ranges)
    all_layers = []
    for s_start, s_end in stage_ranges:
        if s_start < 0 or s_end < 0 or s_start > s_end or s_end >= total_layers:
            raise ValueError(f"Invalid stage range: {s_start}-{s_end}")
        all_layers.extend(range(s_start, s_end + 1))
    print(f"=== SA-HTN Multi-Stage Pilot ===")
    print(f"Stages: {stage_ranges}")
    print(f"All compressed layers: {all_layers}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ================= 节点 1：评测 原模型 PPL =================
    ppl_original = None
    if args.eval_original:
        if accelerator.is_main_process:
            print("\n[Checkpoint 1/4] Evaluating Original Model PPL...")
        ppl_original = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        if accelerator.is_main_process:
            print(f"🎯 [Result 1/4] Original PPL: {ppl_original}")

    # 1. 全局 ASVD 校准（一次性）
    scales = get_activation_scales(student, tokenizer, num_samples=args.calib_samples, max_len=args.calib_max_len)

    # 2. 逐个 stage 处理 Tucker 压缩
    perm_stages = {}
    for stage_idx, (s_start, s_end) in enumerate(stage_ranges):
        stage_layers = list(range(s_start, s_end + 1))
        if accelerator.is_main_process:
            print(f"\n=== Stage {stage_idx}: layers {stage_layers} ===")
        orig = extract_stage_weights(student, stage_layers)
        scaled = apply_asvd_scaling(orig, scales, stage_layers)

        if args.disable_perm:
            perm = None
            inv_perm = None
            permuted = {
                "gate_perm": scaled["gate"],
                "up_perm": scaled["up"],
                "down_perm": scaled["down"],
            }
        else:
            perm, inv_perm = get_stage_permutation(scaled, stage_layers, args.reduced_dim, generator)
            permuted = apply_stage_permutation(scaled, perm)

        perm_stages[stage_idx] = {
            "layers": stage_layers,
            "perm": perm.cpu() if perm is not None else None,
            "inv_perm": inv_perm.cpu() if inv_perm is not None else None,
        }

        up_tensor = torch.stack([torch.stack([g, u], dim=0) for g, u in zip(permuted["gate_perm"], permuted["up_perm"])], dim=0)
        down_tensor = torch.stack(permuted["down_perm"], dim=0)
        tucker_res = run_stage_tucker(up_tensor, down_tensor, args)

        if inv_perm is None:
            inv_perm = torch.arange(up_tensor.shape[2], dtype=torch.long)

        replace_stage_weights_in_student(student, stage_layers, tucker_res["upward_recon"], tucker_res["downward_recon"], inv_perm, scales)

    # ================= 节点 2：评测 Tucker 压缩后 PPL =================
    ppl_tucker = None
    if args.eval_after_tucker:
        if accelerator.is_main_process:
            print("\n[Checkpoint 2/4] Evaluating Tucker-only Model PPL...")
        ppl_tucker = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        if accelerator.is_main_process:
            print(f"🎯 [Result 2/4] Tucker-only PPL: {ppl_tucker}")

    student_after_tucker_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    # 4. 挂载 LoRA（所有压缩层）
    for idx in all_layers:
        mlp = student.model.layers[idx].mlp
        mlp.gate_proj = TuckerLoRALinear(mlp.gate_proj, args.lora_rank)
        mlp.up_proj = TuckerLoRALinear(mlp.up_proj, args.lora_rank)
        mlp.down_proj = TuckerLoRALinear(mlp.down_proj, args.lora_rank)

    # 5. 逐层恢复
    calib_inputs = build_layerwise_calib_inputs(tokenizer, args.calib_samples, args.calib_max_len)
    run_layerwise_healing(student, teacher, calib_inputs, all_layers, args.layerwise_steps, args.learning_rate, accelerator, save_dir)

    # ================= 节点 3：评测 逐层恢复后 PPL =================
    ppl_post_layerwise = None
    if args.eval_post_layerwise:
        if accelerator.is_main_process:
            print("\n[Checkpoint 3/4] Evaluating Post-Layerwise Healing PPL...")
        ppl_post_layerwise = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        if accelerator.is_main_process:
            print(f"🎯 [Result 3/4] Post-Layerwise PPL: {ppl_post_layerwise}")

    # 6. 全局蒸馏
    if accelerator.is_main_process:
        print("\n[Phase 10] Global E2E Distillation...")
    student = train_healing(
        student_model=student,
        tokenizer=tokenizer,
        teacher_model=teacher,
        dataset_name="mixed",
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.learning_rate,
        seq_len=args.seq_len,
        save_every_n_steps=args.save_every_n_steps,
        checkpoint_dir=args.save_dir,
        max_update_steps=args.distill_steps,
        resume_from_checkpoint=None,
        accelerator=accelerator,
        tune_mpo=False,
    )

    # ================= 节点 4：评测 最终 E2E PPL =================
    if accelerator.is_main_process:
        print("\n[Checkpoint 4/4] Evaluating Final E2E Healing PPL...")
    ppl_final = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
    
    # 8. 参数统计与最终报告
    counts = compute_stage_parameter_counts(
        stage_ranges=stage_ranges,
        d_out=11008,
        d_in=4096,
        rank_layer_up=args.rank_layer_up,
        rank_module_up=args.rank_module_up,
        rank_out_up=args.rank_out_up,
        rank_in_up=args.rank_in_up,
        rank_layer_down=args.rank_layer_down,
        rank_in_down=args.rank_in_down,
        rank_out_down=args.rank_out_down,
        lora_rank=args.lora_rank,
    )

    # 计算整体模型的压缩率
    whole_model_compressed_params = total_params_orig - counts["original_total"] + counts["compressed_total"]
    whole_model_compression_ratio = whole_model_compressed_params / total_params_orig

    metrics = {
        "architecture": {
            "model_name": args.model_name,
            "stages": stage_ranges,
            "all_layers": all_layers,
        },
        "compression_hyperparams": {
            "disable_perm": args.disable_perm,
            "reduced_dim": args.reduced_dim,
            "calib_samples": args.calib_samples,
            "rank_layer_up": args.rank_layer_up,
            "rank_module_up": args.rank_module_up,
            "rank_out_up": args.rank_out_up,
            "rank_in_up": args.rank_in_up,
            "rank_layer_down": args.rank_layer_down,
            "rank_in_down": args.rank_in_down,
            "rank_out_down": args.rank_out_down,
        },
        "training_hyperparams": {
            "lora_rank": args.lora_rank,
            "layerwise_steps_per_layer": args.layerwise_steps,
            "distill_steps": args.distill_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "accum_steps": args.accum_steps,
            "seq_len": args.seq_len,
            "save_every_n_steps": args.save_every_n_steps,
        },
        "results": {
            "ppl_original": ppl_original,
            "ppl_tucker_only": ppl_tucker if args.eval_after_tucker else None,
            "ppl_post_layerwise": ppl_post_layerwise,
            "ppl_final_e2e": ppl_final,
        },
        "parameter_counts": {
            **counts,
            "whole_model_original_params": total_params_orig,
            "whole_model_compressed_params": whole_model_compressed_params,
            "whole_model_compression_ratio": whole_model_compression_ratio,
        },
    }

    if accelerator.is_main_process:
        print("\n" + "="*65)
        print("🚀 终极压缩与评测报告 (FINAL COMPRESSION REPORT) 🚀")
        print("="*65)
        print(f" [模型参数分析]")
        print(f" - 全模型原始参数量       : {total_params_orig / 1e9:.4f} B")
        print(f" - 全模型压缩后参数量     : {whole_model_compressed_params / 1e9:.4f} B")
        print(f" - 🗜️ 全模型参数保留率    : {whole_model_compression_ratio * 100:.2f}% (压缩后/原模型)")
        print(f" - 🎯 目标层级参数保留率  : {counts['compressed_total'] / counts['original_total'] * 100:.2f}% (Tucker截断区局部占比)")
        print("-" * 65)
        print(f" [困惑度演化 (WikiText-2 PPL)]")
        print(f" 1. 原模型基线 (Baseline)         : {ppl_original}")
        print(f" 2. Tucker 静态压缩后             : {ppl_tucker}")
        print(f" 3. 逐层微观修复后 (Layerwise)    : {ppl_post_layerwise}")
        print(f" 4. 全局端到端蒸馏后 (E2E Final)  : {ppl_final}")
        print("="*65 + "\n")

        unwrapped_student = accelerator.unwrap_model(student)
        save_artifacts(save_dir, metrics, perm_stages if len(perm_stages) > 0 else None,
                       student_after_tucker_state, unwrapped_student)
        print(f"📦 All artifacts successfully saved to {save_dir.absolute()}")


if __name__ == "__main__":
    main()\


'''
 [困惑度演化 (WikiText-2 PPL)]
 1. 原模型基线 (Baseline)         : 5.43
 2. Tucker 静态压缩后             : 49.88
 3. 逐层微观修复后 (Layerwise)    : 21.38
 4. 全局端到端蒸馏后 (E2E Final)  : 10.5
 '''