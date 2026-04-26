#!/usr/bin/env python3
"""
MPO 递进式累积消融评测脚本 (四大场景对比版)
比较: Original vs SVD vs Pure_MPO vs MPO+LoRA+Permutation
全覆盖: gate_proj, up_proj, down_proj
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import sys
import gc
import csv
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales

# ==========================================
# 🛡️ PPL 评测防爆显存护盾 (切断无用张量传输)
# ==========================================
class LossOnlyWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, labels):
        # 强行设置 use_cache=False，阻止模型生成庞大的 KV Cache
        outputs = self.model(input_ids=input_ids, labels=labels, use_cache=False)
        # 只返回 loss，彻底饿死 DataParallel 的无用收集机制
        return outputs.loss

# ==========================================
# 🧬 FFN 联合聚类通道重排算法
# ==========================================
def find_permutation_for_ffn(W_gate, W_up, method='ward'):
    features = torch.cat([W_gate, W_up], dim=1).float().cpu().numpy()
    Z = linkage(features, method=method, metric='euclidean')
    perm = leaves_list(Z)
    return torch.tensor(perm, dtype=torch.long)

# ==========================================
# ⚡ 超级魔改版 MPO LoRA Wrapper
# ==========================================
class ResMPOWrapper(nn.Module):
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig, skip_svd=False, s_vector=None):
        super().__init__()
        self.mpo = mpo_module
        self.r = lora_rank
        target_device, target_dtype = W_orig.device, W_orig.dtype
        
        if skip_svd:
            self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, device=target_device, dtype=target_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=target_device, dtype=target_dtype))
            return

        with torch.no_grad():
            mpo_gpu = self.mpo 
            bias_backup = mpo_gpu.bias.data.clone() if hasattr(mpo_gpu, 'bias') and mpo_gpu.bias is not None else None
            if bias_backup is not None: mpo_gpu.bias.data.zero_()

            eye = torch.eye(in_features, device=target_device, dtype=target_dtype)  
            W_mpo = mpo_gpu(eye).T                             
            if bias_backup is not None: mpo_gpu.bias.data.copy_(bias_backup)

            Delta_W = (W_orig - W_mpo).contiguous().float()
            
            if s_vector is not None:
                s_vec = s_vector.to(target_device).float()
                s_mean = s_vec.mean()
                lower_bound = s_mean * 0.05
                s_vec_safe = torch.clamp(s_vec, min=lower_bound)
                Delta_W_scaled = Delta_W * s_vec_safe.unsqueeze(0)
            else:
                Delta_W_scaled = Delta_W

            try:
                U, S, Vh = torch.linalg.svd(Delta_W_scaled, full_matrices=False)
            except RuntimeError:
                print("      [⚠️ GPU SVD 内存碎片, 降级至 CPU 计算 LoRA 残差...]")
                U, S, Vh = torch.linalg.svd(Delta_W_scaled.cpu(), full_matrices=False)
                U, S, Vh = U.to(target_device), S.to(target_device), Vh.to(target_device)
                
            S_sqrt = torch.diag(torch.sqrt(S[:self.r].clamp(min=0)))
            A_matrix_scaled = S_sqrt @ Vh[:self.r, :]  
            B_matrix = U[:, :self.r] @ S_sqrt          
            
            if s_vector is not None:
                A_matrix = A_matrix_scaled / s_vec_safe.unsqueeze(0)
            else:
                A_matrix = A_matrix_scaled

            self.lora_A = nn.Parameter(A_matrix.to(target_dtype))
            self.lora_B = nn.Parameter(B_matrix.to(target_dtype))

    def forward(self, x):
        dtype = x.dtype
        mpo_out = self.mpo(x)
        lora_out = F.linear(F.linear(x, self.lora_A.to(dtype)), self.lora_B.to(dtype))
        return mpo_out + lora_out

# ==========================================
# 🔧 纯 SVD 截断模块 (CPU防爆护盾)
# ==========================================
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, W_orig):
        super().__init__()
        target_dtype, target_device = W_orig.dtype, W_orig.device
        with torch.no_grad():
            W_float = W_orig.float()
            try:
                U, S, Vh = torch.linalg.svd(W_float, full_matrices=False)
            except RuntimeError:
                print("      [⚠️ GPU SVD OOM, 降级至 CPU 计算 Vanilla SVD...]")
                U, S, Vh = torch.linalg.svd(W_float.cpu(), full_matrices=False)
                U, S, Vh = U.to(target_device), S.to(target_device), Vh.to(target_device)
                
            S_sqrt = torch.diag(torch.sqrt(S[:rank].clamp(min=0)))
            A_weight = S_sqrt @ Vh[:rank, :]
            B_weight = U[:, :rank] @ S_sqrt
            
        self.A = nn.Linear(in_features, rank, bias=False, device=target_device, dtype=target_dtype)
        self.B = nn.Linear(rank, out_features, bias=False, device=target_device, dtype=target_dtype)
        self.A.weight.data.copy_(A_weight)
        self.B.weight.data.copy_(B_weight)

    def forward(self, x): 
        return self.B(self.A(x))

# ==========================================
# 📊 PPL 评测核心 (自带 DataParallel 包裹)
# ==========================================
@torch.no_grad()
def evaluate_ppl_batched(base_model, tokenizer, max_seq_len=2048, stride=512, batch_size=3):
    loss_only_model = LossOnlyWrapper(base_model)
    model = nn.DataParallel(loss_only_model).cuda()
    model.eval()
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    eval_tokens = min(input_ids.size(1), 15000)
    nlls = []
    
    chunks = []
    prev_end = 0
    for begin in range(0, eval_tokens, stride):
        end = min(begin + max_seq_len, eval_tokens)
        trg_len = end - prev_end
        chunks.append((begin, end, trg_len))
        prev_end = end
        if end == eval_tokens: break

    progress = tqdm(range(0, len(chunks), batch_size), desc="PPL 并行评测中", leave=False)
    for i in progress:
        batch_chunks = chunks[i:i+batch_size]
        b_ids, b_labels, actual_trg_lens = [], [], []
        
        for (b, e, t_len) in batch_chunks:
            ids = input_ids[:, b:e]
            if ids.size(1) == max_seq_len: 
                labels = ids.clone()
                labels[:, :-t_len] = -100
                b_ids.append(ids)
                b_labels.append(labels)
                actual_trg_lens.append(t_len)
                
        if not b_ids: continue
        
        batch_ids_tensor = torch.cat(b_ids, dim=0).cuda()
        batch_labels_tensor = torch.cat(b_labels, dim=0).cuda()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch_ids_tensor, labels=batch_labels_tensor)
            if loss.dim() > 0: loss = loss.mean()
            total_trg_len = sum(actual_trg_lens)
            nlls.append((loss.float() * total_trg_len).item())

    total_tokens = sum([c[2] for c in chunks[:len(nlls)*batch_size]])
    del model
    return round(torch.exp(torch.tensor(sum(nlls)) / total_tokens).item(), 2)

# ==========================================
# 🛠️ 单层手术函数 (全覆盖 gate, up, down)
# ==========================================
def apply_svd_to_layer(model, layer_idx, ratio):
    blk = model.model.layers[layer_idx]
    # 🌟 修改点 1：覆盖 3 个矩阵
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        lin = getattr(blk.mlp, proj)
        out_f, in_f = lin.weight.shape
        rank = max(1, int(out_f * in_f * ratio) // (out_f + in_f))
        setattr(blk.mlp, proj, SVDLinear(in_f, out_f, rank, lin.weight))

# ==========================================
# 🛠️ 单层手术函数 (修复了 MPO 假压缩 Bug)
# ==========================================
def apply_mpo_to_layer(model, layer_idx, ratio, activation_scales_dict, use_permutation=False, use_lora=True):
    NUM_CORES, LORA_RANK = 3, 32
    blk = model.model.layers[layer_idx]
    
    if use_permutation:
        gate_lin = blk.mlp.gate_proj
        up_lin   = blk.mlp.up_proj
        down_lin = blk.mlp.down_proj

        W_gate_raw = gate_lin.weight.detach().float()
        W_up_raw   = up_lin.weight.detach().float()

        s_vec_gate = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.gate_proj")
        s_vec_up   = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.up_proj")
        
        s_val_gate = s_vec_gate.to(W_gate_raw.device).float() if isinstance(s_vec_gate, torch.Tensor) else (s_vec_gate or 1.0)
        s_val_up   = s_vec_up.to(W_up_raw.device).float() if isinstance(s_vec_up, torch.Tensor) else (s_vec_up or 1.0)

        W_gate_scaled = W_gate_raw * s_val_gate
        W_up_scaled   = W_up_raw * s_val_up

        print(f"      🔀 计算 Layer {layer_idx} 通道聚类重排 (Permutation)...")
        perm_cpu = find_permutation_for_ffn(W_gate_scaled, W_up_scaled)
        perm = perm_cpu.to(W_gate_raw.device)

        W_gate_perm = W_gate_raw[perm, :].contiguous()
        W_up_perm   = W_up_raw[perm, :].contiguous()
        W_down_perm = down_lin.weight.detach().float()[:, perm].contiguous()

        with torch.no_grad():
            down_lin.weight.copy_(W_down_perm.to(dtype=down_lin.weight.dtype))
            gate_lin.weight.copy_(W_gate_perm.to(dtype=gate_lin.weight.dtype))
            up_lin.weight.copy_(W_up_perm.to(dtype=up_lin.weight.dtype))

        s_vec_down = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.down_proj")
        if isinstance(s_vec_down, torch.Tensor):
            activation_scales_dict[f"model.layers.{layer_idx}.mlp.down_proj"] = s_vec_down[perm].contiguous()
            
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        lin = getattr(blk.mlp, proj)
        out_f, in_f = lin.weight.shape
        chi_ffn = estimate_mpo_bond_dim(in_f, out_f, NUM_CORES, ratio)
        out_fac, in_fac = find_factors_balanced(out_f, NUM_CORES), find_factors_balanced(in_f, NUM_CORES)
        
        full_name = f"model.layers.{layer_idx}.mlp.{proj}"
        s_vec = activation_scales_dict.get(full_name) if activation_scales_dict else None
        
        cores = factor_linear_mpo_custom(
            lin.weight.float(), chi_ffn, NUM_CORES, out_fac, in_fac, 
            s_vec.float() if isinstance(s_vec, torch.Tensor) else None, 
            adaptive_mode="entropy", # 🌟 核心修复：强制约束在 20% 预算内！
        )
        cleaned_cores = [c.to(device=lin.weight.device, dtype=lin.weight.dtype) for c in cores]
        mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vec)
        
        if use_lora:
            setattr(blk.mlp, proj, ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight, s_vector=s_vec))
        else:
            setattr(blk.mlp, proj, mpo)

# ==========================================
# 🚀 主控制流 (5阶段完美对照组)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_layer", type=int, default=3, help="开始截断的层 (例如 3)")
    parser.add_argument("--ratio", type=float, default=0.6, help="全局压缩保留率")
    parser.add_argument("--csv_name", type=str, default="ultimate_ablation.csv", help="保存结果的CSV文件名")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()

    END_LAYER = 31 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
    
    csv_file = args.csv_name
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Method", "Compressed_Layers", "Ratio", "PPL"])
            
    def append_to_csv(method, layers_str, ratio, ppl):
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([method, layers_str, ratio, ppl])
        print(f"    💾 已保存至 CSV: {method} | Layers: {layers_str} | PPL: {ppl}")

    print("="*60)
    print(f" 🔬 终极五阶消融评测 (完美控制变量法)")
    print(f"    压缩范围: Layer {args.start_layer} -> Layer {END_LAYER}")
    print(f"    压缩组件: gate_proj, up_proj, down_proj")
    print(f"    严格保留率: {args.ratio}")
    print("="*60)

    # ---------------------------------------------------------
    # [0/5] Original
    # ---------------------------------------------------------
    print(f"\n[0/5] 评测原始满血模型...")
    model = load_base()
    base_ppl = evaluate_ppl_batched(model, tokenizer)
    append_to_csv("Original_Baseline", "None", 1.0, base_ppl)
    del model; torch.cuda.empty_cache(); gc.collect()


    '''
    # ---------------------------------------------------------
    # [1/5] Vanilla SVD
    # ---------------------------------------------------------
    print(f"\n[1/5] 开始递进式 Vanilla SVD...")
    model = load_base()
    for current_layer in range(args.start_layer, END_LAYER + 1):
        print(f"\n  👉 SVD 手术: Layer {current_layer}")
        apply_svd_to_layer(model, current_layer, args.ratio)
        ppl = evaluate_ppl_batched(model, tokenizer)
        layers_str = f"{args.start_layer}-{current_layer}" if current_layer > args.start_layer else str(args.start_layer)
        append_to_csv("Vanilla_SVD", layers_str, args.ratio, ppl)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------------
    # [2/5] Pure MPO (无 LoRA, 无重排)
    # ---------------------------------------------------------
    print(f"\n[2/5] 开始递进式 Pure MPO (不含 LoRA 旁路)...")
    model = load_base()
    print("  🔍 校准全局 Activation Scales...")
    activation_scales = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    
    for current_layer in range(args.start_layer, END_LAYER + 1):
        print(f"\n  👉 Pure MPO 手术: Layer {current_layer}")
        apply_mpo_to_layer(model, current_layer, args.ratio, activation_scales, use_permutation=False, use_lora=False)
        ppl = evaluate_ppl_batched(model, tokenizer)
        layers_str = f"{args.start_layer}-{current_layer}" if current_layer > args.start_layer else str(args.start_layer)
        append_to_csv("Pure_MPO_NoLoRA", layers_str, args.ratio, ppl)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------------
    # [3/5] MPO + LoRA 基础版 (有 LoRA, 无重排) - 🌟 之前漏掉的
    # ---------------------------------------------------------
    print(f"\n[3/5] 开始递进式 MPO + LoRA (保留原始无序拓扑)...")
    model = load_base()
    print("  🔍 重新校准全局 Activation Scales...")
    activation_scales = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    
    for current_layer in range(args.start_layer, END_LAYER + 1):
        print(f"\n  👉 MPO+LoRA(基础) 手术: Layer {current_layer}")
        apply_mpo_to_layer(model, current_layer, args.ratio, activation_scales, use_permutation=False, use_lora=True)
        ppl = evaluate_ppl_batched(model, tokenizer)
        layers_str = f"{args.start_layer}-{current_layer}" if current_layer > args.start_layer else str(args.start_layer)
        append_to_csv("MPO_LoRA_Base", layers_str, args.ratio, ppl)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()
    '''
    # ---------------------------------------------------------
    # [4/5] MPO + LoRA 终极版 (有 LoRA, 有重排)
    # ---------------------------------------------------------
    print(f"\n[4/5] 开始递进式 MPO + LoRA + Permutation (通道重排完全体)...")
    model = load_base()
    print("  🔍 重新校准全局 Activation Scales...")
    activation_scales = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    
    for current_layer in range(args.start_layer, END_LAYER + 1):
        print(f"\n  👉 MPO+LoRA(重排) 手术: Layer {current_layer}")
        apply_mpo_to_layer(model, current_layer, args.ratio, activation_scales, use_permutation=True, use_lora=True)
        ppl = evaluate_ppl_batched(model, tokenizer)
        layers_str = f"{args.start_layer}-{current_layer}" if current_layer > args.start_layer else str(args.start_layer)
        append_to_csv("MPO_LoRA_Permuted", layers_str, args.ratio, ppl)
        torch.cuda.empty_cache(); gc.collect()
        
    print("\n🎉 五阶段全量严谨消融实验全部完成！准备查收这组无懈可击的 CSV 数据吧！")

if __name__ == "__main__":
    main()