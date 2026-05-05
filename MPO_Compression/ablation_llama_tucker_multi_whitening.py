#!/usr/bin/env python3
"""
SA-HTN Multi-Stage Pilot (Whitening + Permutation + Tucker + Healing)
完全体代码：包含强制挂载临时目录、动态能量截断、解耦白化与执行时间统计。
"""

from __future__ import annotations

import os
# ================= 暴力挟持临时文件夹 (解决磁盘爆满问题) =================
os.environ["TMPDIR"] = "/mnt/sx_data/my_tmp"
os.environ["TEMP"] = "/mnt/sx_data/my_tmp"
os.environ["TMP"] = "/mnt/sx_data/my_tmp"
import tempfile
tempfile.tempdir = "/mnt/sx_data/my_tmp"
# ========================================================================

os.environ["HF_HOME"] = "/mnt/hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hf-cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import math
import sys
import time
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

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
from healing import train_healing


# ---------- 命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--stage_ranges", type=str, nargs='+', default=["12-15"])
    parser.add_argument("--save_dir", type=str, default="./outputs/sa_htn_whitening")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--calib_max_len", type=int, default=512)
    parser.add_argument("--reduced_dim", type=int, default=512)
    parser.add_argument("--disable_perm", action="store_true")
    
    # 这里的 Rank 现在作为动态截断的安全上限
    parser.add_argument("--rank_layer_up", type=int, default=4)
    parser.add_argument("--rank_module_up", type=int, default=2)
    parser.add_argument("--rank_layer_down", type=int, default=4)
    parser.add_argument("--rank_out_up", type=int, default=8448)
    parser.add_argument("--rank_in_up", type=int, default=3136)
    parser.add_argument("--rank_in_down", type=int, default=3136)
    parser.add_argument("--rank_out_down", type=int, default=8448)
    
    parser.add_argument("--energy_threshold", type=float, default=0.90, help="保留的奇异值能量比例 (0.0~1.0)")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--layerwise_steps", type=int, default=100)
    parser.add_argument("--distill_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
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


# ---------- 白化矩阵计算 (逐层解耦白化版) ----------
def compute_stage_whitening_matrices(
    model: nn.Module,
    tokenizer,
    calib_input_ids: list[torch.Tensor],
    stage_layers: list[int],
    eps: float = 1e-4,
) -> dict:
    device = next(model.parameters()).device

    dummy_mlp = model.model.layers[stage_layers[0]].mlp
    d_in = dummy_mlp.gate_proj.weight.shape[1]      
    d_inter = dummy_mlp.gate_proj.weight.shape[0]   

    covs = {
        idx: {
            "hidden": torch.zeros((d_in, d_in), device=device, dtype=torch.float64),
            "inter": torch.zeros((d_inter, d_inter), device=device, dtype=torch.float64),
            "tokens": 0
        } for idx in stage_layers
    }

    def make_hook(layer_idx, proj_type):
        def hook(module, args):
            x = args[0].detach().view(-1, args[0].shape[-1]).to(torch.float64)
            covs[layer_idx][proj_type].addmm_(x.t(), x)
        return hook

    hooks = []
    for idx in stage_layers:
        mlp = model.model.layers[idx].mlp
        hooks.append(mlp.gate_proj.register_forward_pre_hook(make_hook(idx, "hidden")))
        hooks.append(mlp.down_proj.register_forward_pre_hook(make_hook(idx, "inter")))

    with torch.no_grad():
        for input_ids in tqdm(calib_input_ids, desc="计算逐层白化协方差", leave=False):
            num_t = input_ids.numel()
            for idx in stage_layers:
                covs[idx]["tokens"] += num_t
            model(input_ids.to(device))

    for h in hooks:
        h.remove()

    def get_robust_whitening_transforms(C: torch.Tensor, eps: float):
        n = C.shape[0]
        C = C + eps * torch.eye(n, device=C.device, dtype=C.dtype)
        if not torch.isfinite(C).all():
            C = torch.where(torch.isfinite(C), C, torch.zeros_like(C))
            C = C + eps * torch.eye(n, device=C.device, dtype=C.dtype)
        try:
            L = torch.linalg.cholesky(C)
        except torch.linalg.LinAlgError:
            diag_max = C.diagonal().abs().max().item()
            for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                reg = max(diag_max * scale, 1e-3)
                C_reg = C + reg * torch.eye(n, device=C.device, dtype=C.dtype)
                try:
                    L = torch.linalg.cholesky(C_reg)
                    break
                except torch.linalg.LinAlgError:
                    continue
            else:
                C = torch.eye(n, device=C.device, dtype=C.dtype)
                L = torch.linalg.cholesky(C)
        L_inv = torch.linalg.inv(L)
        return L, L_inv 

    whiten_mats = {}
    for idx in stage_layers:
        C_hidden = (covs[idx]["hidden"] / covs[idx]["tokens"]).float()
        C_inter = (covs[idx]["inter"] / covs[idx]["tokens"]).float()
        
        h_L, h_L_inv = get_robust_whitening_transforms(C_hidden, eps)
        i_L, i_L_inv = get_robust_whitening_transforms(C_inter, eps)
        
        whiten_mats[idx] = {
            "hidden_L": h_L,
            "hidden_L_inv": h_L_inv,
            "intermediate_L": i_L,
            "intermediate_L_inv": i_L_inv,
        }

    return whiten_mats


# ---------- 权重提取 ----------
def extract_stage_weights(model, layer_indices):
    gate_w, up_w, down_w = [], [], []
    for idx in layer_indices:
        mlp = model.model.layers[idx].mlp
        gate_w.append(mlp.gate_proj.weight.detach().clone())
        up_w.append(mlp.up_proj.weight.detach().clone())
        down_w.append(mlp.down_proj.weight.detach().clone())
    return {"gate": gate_w, "up": up_w, "down": down_w}


# ---------- 白化变换 ----------
def apply_whitening(stage_weights, whiten_mats, layer_indices):
    white_gate, white_up, white_down = [], [], []
    for idx in layer_indices:
        h_L = whiten_mats[idx]["hidden_L"]
        i_L = whiten_mats[idx]["intermediate_L"]
        
        w_g = stage_weights["gate"].pop(0)
        white_gate.append(w_g.float() @ h_L)
        del w_g
        
        w_u = stage_weights["up"].pop(0)
        white_up.append(w_u.float() @ h_L)
        del w_u
        
        w_d = stage_weights["down"].pop(0)
        white_down.append(w_d.float() @ i_L)
        del w_d
        
    torch.cuda.empty_cache()
    return {"gate": white_gate, "up": white_up, "down": white_down}


# ---------- 逆白化 + 逆排列 ----------
def replace_stage_weights_in_student(model, layer_indices, recon_up, recon_down, inv_perm, whiten_mats):
    for offset, idx in enumerate(layer_indices):
        hidden_L_inv = whiten_mats[idx]["hidden_L_inv"]
        inter_L_inv  = whiten_mats[idx]["intermediate_L_inv"]
        device = hidden_L_inv.device

        gate_recon = recon_up[offset, 0][inv_perm, :]
        up_recon   = recon_up[offset, 1][inv_perm, :]
        down_recon = recon_down[offset][:, inv_perm]
        
        mlp = model.model.layers[idx].mlp
        with torch.no_grad():
            gate_final = (gate_recon.to(device) @ hidden_L_inv).to(mlp.gate_proj.weight.dtype)
            mlp.gate_proj.weight.copy_(gate_final)
            
            up_final = (up_recon.to(device) @ hidden_L_inv).to(mlp.up_proj.weight.dtype)
            mlp.up_proj.weight.copy_(up_final)
            
            down_final = (down_recon.to(device) @ inter_L_inv).to(mlp.down_proj.weight.dtype)
            mlp.down_proj.weight.copy_(down_final)


# ---------- 排列 ----------
def get_stage_permutation(white_weights, layer_indices, reduced_dim, generator):
    sketches = []
    for gate_s, up_s in zip(white_weights["gate"], white_weights["up"]):
        features = torch.cat([gate_s, up_s], dim=1)
        proj = torch.randn(features.shape[1], reduced_dim, generator=generator, device=features.device, dtype=features.dtype)
        proj /= math.sqrt(reduced_dim)
        sketches.append(features @ proj)
    stage_sketch = torch.stack(sketches, dim=0).mean(dim=0)
    linkage_matrix = linkage(stage_sketch.detach().cpu().numpy(), method="ward", metric="euclidean")
    perm = torch.tensor(leaves_list(linkage_matrix), dtype=torch.long)
    inv_perm = torch.argsort(perm)
    return perm, inv_perm


def apply_stage_permutation(white_weights, perm):
    gate_perm, up_perm, down_perm = [], [], []
    while white_weights["gate"]:
        w = white_weights["gate"].pop(0)
        gate_perm.append(w[perm, :].clone().cpu())
        del w
    while white_weights["up"]:
        w = white_weights["up"].pop(0)
        up_perm.append(w[perm, :].clone().cpu())
        del w
    while white_weights["down"]:
        w = white_weights["down"].pop(0)
        down_perm.append(w[:, perm].clone().cpu())
        del w
    torch.cuda.empty_cache()
    return {
        "gate_perm": gate_perm,
        "up_perm": up_perm,
        "down_perm": down_perm,
    }


# ---------- 动态能量截断 ----------
def get_dynamic_ranks(tensor, energy_threshold=0.95, max_ranks=None):
    ranks = []
    for mode in range(tensor.ndim):
        unfolded = tl.unfold(tensor, mode)
        S = torch.linalg.svdvals(unfolded.float())
        energy = S ** 2
        total_energy = energy.sum()
        cum_energy = torch.cumsum(energy, dim=0)
        
        target = total_energy * energy_threshold
        valid_indices = torch.where(cum_energy >= target)[0]
        
        if len(valid_indices) > 0:
            r = valid_indices[0].item() + 1
        else:
            r = len(S)
            
        if max_ranks and mode < len(max_ranks):
            r = min(r, max_ranks[mode])
            
        ranks.append(max(1, r))
    return ranks


# ---------- Tucker 分解 (动态秩 + HOOI) ----------
def run_stage_tucker(upward_tensor, downward_tensor, args):
    tl.set_backend("pytorch")
    
    max_up_ranks = [args.rank_layer_up, args.rank_module_up, args.rank_out_up, args.rank_in_up]
    max_down_ranks = [args.rank_layer_down, args.rank_in_down, args.rank_out_down]
    
    print(f"  -> 🔍 正在基于能量阈值 ({args.energy_threshold * 100}%) 计算动态秩...")
    upward_ranks = get_dynamic_ranks(upward_tensor, energy_threshold=args.energy_threshold, max_ranks=max_up_ranks)
    downward_ranks = get_dynamic_ranks(downward_tensor, energy_threshold=args.energy_threshold, max_ranks=max_down_ranks)
    
    print(f"  -> ✅ 实际采用的 Tucker Up Ranks:   {upward_ranks}")
    print(f"  -> ✅ 实际采用的 Tucker Down Ranks: {downward_ranks}")

    up_core, up_factors = tucker(
        upward_tensor.float(), 
        rank=upward_ranks, 
        init="svd", 
        n_iter_max=30,  
        tol=1e-4        
    )
    
    down_core, down_factors = tucker(
        downward_tensor.float(), 
        rank=downward_ranks, 
        init="svd", 
        n_iter_max=30,  
        tol=1e-4
    )
    
    return {
        "upward_recon": tl.tucker_to_tensor((up_core, up_factors)),
        "downward_recon": tl.tucker_to_tensor((down_core, down_factors)),
        "actual_up_ranks": upward_ranks,      # 👈 新增：传出真实 Up 秩
        "actual_down_ranks": downward_ranks,  # 👈 新增：传出真实 Down 秩
    }


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


# ---------- 构建校准输入 ----------
def build_layerwise_calib_inputs(tokenizer, calib_samples: int, calib_max_len: int, seed: int = 42):
    import random
    print(f"\n📚 正在加载 WikiText-2 校准数据集 (严格对齐 SVD-LLM)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
    random.seed(seed)
    samples = []
    max_start = max(0, len(tokens) - calib_max_len - 1)
    for _ in range(calib_samples):
        start = random.randint(0, max_start)
        sample = tokens[start : start + calib_max_len].unsqueeze(0)
        samples.append(sample)
    print(f"✅ 成功提取 {len(samples)} 条长度为 {calib_max_len} 的连续校准窗口。")
    return samples


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
    stage_ranges, d_out, d_in,
    rank_layer_up, rank_module_up, rank_out_up, rank_in_up,
    rank_layer_down, rank_in_down, rank_out_down, lora_rank,
):
    original_up = 0
    original_down = 0
    compressed_up = 0
    compressed_down = 0
    lora_params = 0
    for s_start, s_end in stage_ranges:
        L = s_end - s_start + 1
        original_up += L * 2 * d_out * d_in
        original_down += L * d_in * d_out
        compressed_up += (
            rank_layer_up * rank_module_up * rank_out_up * rank_in_up
            + L * rank_layer_up + 2 * rank_module_up
            + d_out * rank_out_up + d_in * rank_in_up
        )
        compressed_down += (
            rank_layer_down * rank_in_down * rank_out_down
            + L * rank_layer_down + d_in * rank_in_down + d_out * rank_out_down
        )
        lora_params += L * 3 * lora_rank * (d_in + d_out)
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

def save_artifacts(save_dir, metrics, perm_stages, student_after_tucker_state, student_final_model):
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
    total_params_orig = sum(p.numel() for p in student.parameters())

    stage_ranges = parse_stage_ranges(args.stage_ranges)
    all_layers = []
    for s_start, s_end in stage_ranges:
        if s_start < 0 or s_end < 0 or s_start > s_end or s_end >= total_layers:
            raise ValueError(f"Invalid stage range: {s_start}-{s_end}")
        all_layers.extend(range(s_start, s_end + 1))
    print(f"=== SA-HTN Multi-Stage Pilot (Whitening) ===")
    print(f"Stages: {stage_ranges}")
    print(f"All compressed layers: {all_layers}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_original:
        if accelerator.is_main_process:
            print("\n[Checkpoint 1/4] Evaluating Original Model PPL...")
        ppl_original = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        if accelerator.is_main_process:
            print(f"🎯 [Result 1/4] Original PPL: {ppl_original}")

    calib_inputs = build_layerwise_calib_inputs(
        tokenizer, 
        args.calib_samples, 
        args.calib_max_len, 
        args.seed
    )
    perm_stages = {}
    for stage_idx, (s_start, s_end) in enumerate(stage_ranges):
        stage_layers = list(range(s_start, s_end + 1))
        print(f"\n=== Stage {stage_idx}: layers {stage_layers} ===")

        whiten_mats = compute_stage_whitening_matrices(
            student, tokenizer, calib_inputs, stage_layers, eps=1e-4
        )

        orig = extract_stage_weights(student, stage_layers)
        white = apply_whitening(orig, whiten_mats, stage_layers)
        del orig 

        if args.disable_perm:
            perm = None
            inv_perm = None
            permuted = {
                "gate_perm": white["gate"],
                "up_perm": white["up"],
                "down_perm": white["down"],
            }
        else:
            perm, inv_perm = get_stage_permutation(white, stage_layers, args.reduced_dim, generator)
            permuted = apply_stage_permutation(white, perm)
            del white 

        perm_stages[stage_idx] = {
            "layers": stage_layers,
            "perm": perm.cpu() if perm is not None else None,
            "inv_perm": inv_perm.cpu() if inv_perm is not None else None,
        }

        up_tensor = torch.stack([torch.stack([g, u], dim=0) for g, u in zip(permuted["gate_perm"], permuted["up_perm"])], dim=0)
        down_tensor = torch.stack(permuted["down_perm"], dim=0)

        del permuted
        torch.cuda.empty_cache()
        gc.collect()

        tucker_res = run_stage_tucker(up_tensor, down_tensor, args)


        actual_up_ranks = tucker_res["actual_up_ranks"]
        actual_down_ranks = tucker_res["actual_down_ranks"]


        del up_tensor, down_tensor
        torch.cuda.empty_cache()

        replace_stage_weights_in_student(student, stage_layers, tucker_res["upward_recon"], tucker_res["downward_recon"], inv_perm, whiten_mats)
        
        del whiten_mats, tucker_res
        torch.cuda.empty_cache()
        gc.collect()

    ppl_tucker = None
    if args.eval_after_tucker:
        if accelerator.is_main_process:
            print("\n[Checkpoint 2/4] Evaluating Tucker-only Model PPL...")
        ppl_tucker = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        if accelerator.is_main_process:
            print(f"🎯 [Result 2/4] Tucker-only PPL: {ppl_tucker}")

    student_after_tucker_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    for idx in all_layers:
        mlp = student.model.layers[idx].mlp
        mlp.gate_proj = TuckerLoRALinear(mlp.gate_proj, args.lora_rank)
        mlp.up_proj = TuckerLoRALinear(mlp.up_proj, args.lora_rank)
        mlp.down_proj = TuckerLoRALinear(mlp.down_proj, args.lora_rank)

    run_layerwise_healing(student, teacher, calib_inputs, all_layers, args.layerwise_steps, args.learning_rate, accelerator, save_dir)

    ppl_post_layerwise = None
    if args.eval_post_layerwise:
        if accelerator.is_main_process:
            print("\n[Checkpoint 3/4] Evaluating Post-Layerwise Healing PPL...")
        ppl_post_layerwise = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
        if accelerator.is_main_process:
            print(f"🎯 [Result 3/4] Post-Layerwise PPL: {ppl_post_layerwise}")
    
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

    ppl_final = eval_ppl(student, tokenizer, stride=args.eval_stride, max_tokens=args.eval_max_tokens)
    if accelerator.is_main_process:
        print(f"🎯 [Result 4/4] Final PPL: {ppl_final}")
    

    # 使用动态计算出的真实 Rank 算账
    counts = compute_stage_parameter_counts(
        stage_ranges=stage_ranges,
        d_out=11008, d_in=4096,
        rank_layer_up=actual_up_ranks[0], rank_module_up=actual_up_ranks[1],
        rank_out_up=actual_up_ranks[2], rank_in_up=actual_up_ranks[3],
        rank_layer_down=actual_down_ranks[0], rank_in_down=actual_down_ranks[1],
        rank_out_down=actual_down_ranks[2], lora_rank=args.lora_rank,
    )
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
            "ppl_tucker_only": ppl_tucker,
            "ppl_post_layerwise": ppl_post_layerwise,
            #"ppl_final_e2e": ppl_final,
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
        print("🚀 终极压缩与评测报告 (FINAL COMPRESSION REPORT) [Whitening] 🚀")
        print("="*65)
        print(f" [模型参数分析]")
        print(f" - 全模型原始参数量        : {total_params_orig / 1e9:.4f} B")
        print(f" - 全模型压缩后参数量      : {whole_model_compressed_params / 1e9:.4f} B")
        print(f" - 🗜️ 全模型参数保留率    : {whole_model_compression_ratio * 100:.2f}%")
        print(f" - 🎯 目标层级参数保留率  : {counts['compressed_total'] / counts['original_total'] * 100:.2f}%")
        print("-" * 65)
        print(f" [困惑度演化 (WikiText-2 PPL)]")
        print(f" 1. 原模型基线 (Baseline)         : {ppl_original}")
        print(f" 2. Tucker 静态压缩后             : {ppl_tucker}")
        print(f" 3. 逐层微观修复后 (Layerwise)    : {ppl_post_layerwise}")
        print(f" 4. 全局端到端蒸馏后 (E2E Final)  : {ppl_final}")
        print("="*65 + "\n")

        save_artifacts(save_dir, metrics, perm_stages if len(perm_stages) > 0 else None,
                       student_after_tucker_state, accelerator.unwrap_model(student))
        print(f"📦 All artifacts successfully saved to {save_dir.absolute()}")


if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Total execution time: {elapsed / 60:.2f} minutes")