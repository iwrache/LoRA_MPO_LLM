#!/usr/bin/env python3
"""
MPO 多层极限消融评测脚本 (Ablation Study)
用法示例:
python run_custom_multi_ablation.py \
    --layers 16 18 20 \
    --ratios 0.2 0.3 0.15 \
    --healed_ckpt ./custom_layers_healed.pt
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys
import gc
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales

# 导入你在 main_custom_multi_layers.py 里写好的 Wrapper
from main_custom_multi_layers import ResMPOWrapper 

# ==========================================
# 🔧 纯 SVD 截断模块
# ==========================================
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, W_orig):
        super().__init__()
        target_dtype, target_device = W_orig.dtype, W_orig.device
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_orig.float(), full_matrices=False)
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
# 📊 PPL 评测核心
# ==========================================
@torch.no_grad()
def evaluate_ppl(model, tokenizer, max_seq_len=2048, stride=512):
    model.eval()
    device = model.device if hasattr(model, 'device') else torch.device("cuda:0")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    eval_tokens = min(input_ids.size(1), 15000) # 测1.5万个Token
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
# 🛠️ 骨架搭建辅助函数 (为了复用)
# ==========================================
def build_mpo_skeleton(model, layers, ratios, activation_scales_dict):
    """为传入的模型，将指定的多层替换为 MPO 骨架"""
    NUM_CORES, LORA_RANK = 3, 32
    for layer_idx, ratio in zip(layers, ratios):
        blk = model.model.layers[layer_idx]
        for proj in ["gate_proj", "up_proj"]:
            lin = getattr(blk.mlp, proj)
            out_f, in_f = lin.weight.shape
            chi_ffn = estimate_mpo_bond_dim(in_f, out_f, NUM_CORES, ratio)
            out_fac, in_fac = find_factors_balanced(out_f, NUM_CORES), find_factors_balanced(in_f, NUM_CORES)
            
            full_name = f"model.layers.{layer_idx}.mlp.{proj}"
            s_vec = activation_scales_dict.get(full_name) if activation_scales_dict else None
            
            cores = factor_linear_mpo_custom(
                lin.weight.float(), chi_ffn, NUM_CORES, out_fac, in_fac, 
                s_vec.float() if s_vec is not None else None, 
                adaptive=True, energy_threshold=0.99
            )
            cleaned_cores = [c.to(device=lin.weight.device, dtype=lin.weight.dtype) for c in cores]
            mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vec)
            
            # 挂载 Wrapper
            setattr(blk.mlp, proj, ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs='+', default=[14,15,16,17,18])
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.2,0.2,0.2,0.2,0.2])
    parser.add_argument("--healed_ckpt", type=str, default="/mnt/sx_data")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()

    if len(args.layers) != len(args.ratios):
        raise ValueError("❌ layers 和 ratios 的列表长度必须完全一致！")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = {}

    def load_base():
        return AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    print("="*60)
    print(f" 🔬 多层极限消融评测")
    print(f"    目标层: {args.layers}")
    print(f"    保留率: {args.ratios}")
    print("="*60)

    # ---------------------------------------------------------
    # 1. Original
    # ---------------------------------------------------------
    print(f"\n[1/4] 评测原始满血模型...")
    model = load_base()
    results["1. Original"] = evaluate_ppl(model, tokenizer)
    del model; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------------
    # 2. Vanilla SVD (多层)
    # ---------------------------------------------------------
    print(f"\n[2/4] 评测 Vanilla SVD 截断 (替换多层)...")
    model = load_base()
    for layer_idx, ratio in zip(args.layers, args.ratios):
        blk = model.model.layers[layer_idx]
        for proj in ["gate_proj", "up_proj"]:
            lin = getattr(blk.mlp, proj)
            out_f, in_f = lin.weight.shape
            rank = max(1, int(out_f * in_f * ratio) // (out_f + in_f))
            setattr(blk.mlp, proj, SVDLinear(in_f, out_f, rank, lin.weight))
    results["2. Vanilla SVD"] = evaluate_ppl(model, tokenizer)
    del model; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------------
    # 3. MPO (No Healing, 多层)
    # ---------------------------------------------------------
    print(f"\n[3/4] 评测 MPO+LoRA 初始态 (替换多层)...")
    model = load_base()
    activation_scales = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    
    # 使用辅助函数搭建所有指定层的 MPO 骨架
    model = build_mpo_skeleton(model, args.layers, args.ratios, activation_scales)
    results["3. MPO (No Healing)"] = evaluate_ppl(model, tokenizer)
    '''
    # ---------------------------------------------------------
    # 4. MPO (Healed, 多层)
    # ---------------------------------------------------------
    print(f"\n[4/4] 评测 MPO+LoRA 缝合后 (加载 Checkpoint)...")
    # 因为上一步 (阶段 3) 已经把骨架搭好了，我们无需从头再搭！
    # 直接把你的 Checkpoint 灌进这个搭好骨架的模型里！
    ckpt = torch.load(args.healed_ckpt)
    model.load_state_dict(ckpt, strict=False)
    results["4. MPO (Healed)"] = evaluate_ppl(model, tokenizer)
    del model; torch.cuda.empty_cache(); gc.collect()
    '''
    # ---------------------------------------------------------
    # 打印战报
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f" 🚀 多层联合消融战报 (共 {len(args.layers)} 层)")
    print("="*60)
    for k, v in results.items(): print(f"  {k:<25} : {v:>8.2f}")
    print("="*60)

if __name__ == "__main__":
    main()