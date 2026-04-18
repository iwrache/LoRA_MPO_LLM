#!/usr/bin/env python3
"""
MPO 单层极限消融实验 - 评测脚本 (专供 Layer 16)
只对比第 16 层在不同状态下的 Wikitext-2 PPL。
"""
import os
import sys
from pathlib import Path

# ==========================================
# 💡 核心魔法：动态添加父目录到环境变量
# ==========================================
# 1. 获取当前脚本所在的绝对路径目录
current_dir = Path(__file__).resolve().parent

# 2. 获取上一级目录（也就是 test_MPO.py 所在的那个外层文件夹）
parent_dir = current_dir.parent

# 3. 把上一级目录强行塞进 Python 的搜索路径列表的最前面 (index 0)
sys.path.insert(0, str(parent_dir))

# ==========================================
# 现在你可以毫无阻碍地导入外层文件夹里的东西了！
# ==========================================
import test_MPO

import os
import gc
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# 导入你的底层函数
from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales
from main_single_layer_4gpu import ResMPOWrapper # 复用你刚才脚本里的 Wrapper

# ==========================================
# 🔧 纯 SVD 截断模块
# ==========================================
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, W_orig):
        super().__init__()
        target_dtype = W_orig.dtype
        target_device = W_orig.device
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_orig.float().cpu(), full_matrices=False)
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
# 📊 PPL 评测核心 (滑动窗口)
# ==========================================
@torch.no_grad()
def evaluate_ppl(model, tokenizer, max_seq_len=2048, stride=512):
    model.eval()
    device = model.device if hasattr(model, 'device') else torch.device("cuda:0")
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    eval_tokens = min(input_ids.size(1), 15000) # 测1.5万个token，快速出结果
    nlls = []
    prev_end = 0
    
    progress = tqdm(range(0, eval_tokens, stride), desc="PPL 评测中")
    for begin in progress:
        end = min(begin + max_seq_len, eval_tokens)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(ids, labels=labels)
            nlls.append((outputs.loss.float() * trg_len).item())
        prev_end = end
        if end == eval_tokens: break
            
    return round(torch.exp(torch.tensor(sum(nlls)) / prev_end).item(), 2)

# ==========================================
# 🚀 主评测管线
# ==========================================
def main():
    MODEL_NAME = "NousResearch/Llama-2-7b-hf"
    HEALED_CKPT = "./ablation_layer16_only_r0.2.pt" # 你刚才保存的单层模型
    TARGET_LAYER = 16
    TARGET_RATIO = 0.2
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = {}

    def load_base():
        # 单张卡足以跑评测
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    print("="*60)
    print(f" 🔬 单层极限消融实验评测 (仅靶向 Layer {TARGET_LAYER})")
    print("="*60)

    # 1. Original
    print("\n[1/4] 评测原始满血模型...")
    model = load_base()
    results["1. Original"] = evaluate_ppl(model, tokenizer)
    del model; torch.cuda.empty_cache(); gc.collect()

    # 2. Vanilla SVD (仅 Layer 16)
    print("\n[2/4] 评测 Vanilla SVD 截断 (仅 Layer 16)...")
    model = load_base()
    blk = model.model.layers[TARGET_LAYER]
    for proj in ["gate_proj", "up_proj"]:
        lin = getattr(blk.mlp, proj)
        out_f, in_f = lin.weight.shape
        rank = max(1, int(out_f * in_f * TARGET_RATIO) // (out_f + in_f))
        setattr(blk.mlp, proj, SVDLinear(in_f, out_f, rank, lin.weight))
    results["2. Vanilla SVD (Layer 16)"] = evaluate_ppl(model, tokenizer)
    del model; torch.cuda.empty_cache(); gc.collect()

    # 3. MPO (No Healing, 仅 Layer 16)
    print("\n[3/4] 评测 MPO+LoRA 刚切割后的状态 (No Healing)...")
    model = load_base()
    activation_scales = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    blk = model.model.layers[TARGET_LAYER]
    for proj in ["gate_proj", "up_proj"]:
        lin = getattr(blk.mlp, proj)
        out_f, in_f = lin.weight.shape
        chi_ffn = estimate_mpo_bond_dim(in_f, out_f, 3, TARGET_RATIO)
        out_fac, in_fac = find_factors_balanced(out_f, 3), find_factors_balanced(in_f, 3)
        s_vec = activation_scales.get(f"model.layers.{TARGET_LAYER}.mlp.{proj}")
        
        cores = factor_linear_mpo_custom(
            lin.weight.cpu().float(), chi_ffn, 3, out_fac, in_fac, 
            s_vec.cpu().float() if s_vec is not None else None, 
            adaptive=True, energy_threshold=0.99
        )
        cleaned_cores = [c.to(device=lin.weight.device, dtype=lin.weight.dtype) for c in cores]
        mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vec)
        setattr(blk.mlp, proj, ResMPOWrapper(mpo, in_f, out_f, 32, lin.weight))
    results["3. MPO (No Healing)"] = evaluate_ppl(model, tokenizer)
    
    # 4. MPO (Healed, 仅 Layer 16)
    print("\n[4/4] 评测 MPO+LoRA 微调后的状态 (Healed)...")
    # 因为我们上一步已经搭好骨架了，直接把 pt 权重灌进去即可！
    ckpt = torch.load(HEALED_CKPT, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    results["4. MPO (Healed)"] = evaluate_ppl(model, tokenizer)
    del model; torch.cuda.empty_cache(); gc.collect()

    # 输出战报
    print("\n" + "="*60)
    print(f" 🚀 单层 (Layer {TARGET_LAYER}) 极限消融战报 (压缩率: {TARGET_RATIO*100}%)")
    print("="*60)
    for k, v in results.items():
        print(f"  {k:<30} : {v:>8.2f}")
    print("="*60)

if __name__ == "__main__":
    main()