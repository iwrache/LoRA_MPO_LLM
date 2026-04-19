#!/usr/bin/env python3
import os
# 强制使用 4 张卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
from healing import train_healing


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, W_orig):
        super().__init__()
        self.rank = rank
        target_device = W_orig.device
        target_dtype = W_orig.dtype
        
        with torch.no_grad():
            W_f32 = W_orig.to(dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            U_trunc, S_trunc, Vh_trunc = U[:, :rank], S[:rank], Vh[:rank, :]
            S_sqrt = torch.diag(torch.sqrt(S_trunc))
            W_A = S_sqrt @ Vh_trunc
            W_B = U_trunc @ S_sqrt
            
        self.lora_A = nn.Parameter(W_A.to(target_dtype))
        self.lora_B = nn.Parameter(W_B.to(target_dtype))

    def forward(self, x):
        return F.linear(F.linear(x, self.lora_A), self.lora_B)

def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=15000):
    model.eval() 
    print("⏳ 正在计算 Wikitext-2 PPL...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    effective_max = min(max_len, max_ctx)
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    nlls = []
    for begin in tqdm(range(0, seq_len, stride), desc="PPL 评测中"):
        end = min(begin + effective_max, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1: continue
        with torch.no_grad():
            out = model(chunk)
            lm_logits = out.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean"
            )
            if not torch.isnan(loss) and not torch.isinf(loss):
                nlls.append(loss.item())

    if not nlls: return float("nan")
    return round(torch.exp(torch.tensor(nlls).mean()).item(), 2)

def get_u_shape_ratio(layer_idx, total_layers, target_ratio):
    if layer_idx < 3 or layer_idx >= total_layers - 3: return 1.0
    start_idx, end_idx = 3, total_layers - 4
    center, half_range = (start_idx + end_idx) / 2.0, (end_idx - start_idx) / 2.0
    x = (layer_idx - center) / half_range
    base, amplitude = max(0.1, target_ratio - 0.15), 0.45 
    return max(0.1, min(0.95, base + amplitude * (x ** 2)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--target_ratio", type=float, default=0.6)
    # 🚨 新增：支持自定义层的压缩率
    parser.add_argument("--custom_layers", type=int, nargs='*', default=[], help="需自定义保留率的层号")
    parser.add_argument("--custom_ratios", type=float, nargs='*', default=[], help="对应的保留率")
    parser.add_argument("--e2e_steps", type=int, default=1000)
    args = parser.parse_args()

    custom_ratio_map = dict(zip(args.custom_layers, args.custom_ratios))

    print("="*70)
    print(" 🚀 严格参数对齐: LLaMA-2 纯 SVD 压缩基准线测试 (带 E2E Healing)")
    if custom_ratio_map: print(f" 🎯 注入了自定义层压缩率: {custom_ratio_map}")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    print("📦 加载模型...")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
    student_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
    teacher_model.eval()
    
    num_layers = len(student_model.model.layers)
    total_original_params = 0
    total_compressed_params = 0

    print("\n🦴 正在执行 SVD 截断并冻结非必要参数...")
    
    for layer_idx in range(num_layers):
        # 🚨 优先读取自定义比例，如果没有再使用 U 型公式
        ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
        
        if ratio >= 0.99: 
            continue
            
        layer = student_model.model.layers[layer_idx]
        for proj in ["gate_proj", "up_proj"]:
            lin = getattr(layer.mlp, proj)
            W_orig = lin.weight.data
            out_f, in_f = W_orig.shape
            
            orig_params = out_f * in_f
            target_params = int(orig_params * ratio)
            
            rank = max(1, target_params // (in_f + out_f))
            actual_params = rank * (in_f + out_f)
            
            total_original_params += orig_params
            total_compressed_params += actual_params
            
            svd_layer = SVDLinear(in_f, out_f, rank, W_orig)
            setattr(layer.mlp, proj, svd_layer)
            
            # 打印被施加极端保留率的层，确认是否生效
            if layer_idx in custom_ratio_map and proj == "gate_proj":
                 print(f"    🎯 [特殊处理] Layer {layer_idx} 目标保留率: {ratio*100:.1f}%, 实际分配 Rank: {rank}")

    print("\n📊 宏观参数统计:")
    print(f"    - MLP (Gate/Up) 原总参数: {total_original_params/1e6:.2f} M")
    print(f"    - SVD 压缩后总参数:       {total_compressed_params/1e6:.2f} M")
    if total_original_params > 0:
        print(f"    - 实际参数保留率:         {total_compressed_params/total_original_params*100:.2f}%")

    print("\n[阶段 1] 测量未 Healing 的 SVD 初始 PPL...")
    ppl_zero = eval_ppl(student_model, tokenizer)
    print(f"🌟 SVD Zero-Shot PPL: {ppl_zero}")

    print("\n[阶段 2] 🌌 启动 SVD 全局端到端联合蒸馏 (同等条件 Healing)...")
    torch.cuda.empty_cache(); gc.collect()
    
    # 将模型转回 bfloat16 进行加速训练和防爆盘
    student_model.to(torch.bfloat16)
    teacher_model.to(torch.bfloat16)

    student_model = train_healing(
        student_model=student_model, tokenizer=tokenizer, teacher_model=teacher_model, 
        dataset_name="mixed", epochs=1, batch_size=4, accum_steps=2, lr=2e-5, seq_len=1024,  
        save_every_n_steps=1000, checkpoint_dir="./svd_healing_checkpoints", 
        max_update_steps=args.e2e_steps, resume_from_checkpoint=None 
    )

    print("\n[阶段 3] 测量 Healing 后的 SVD PPL...")
    ppl_healed = eval_ppl(student_model, tokenizer)
    
    print("\n" + "="*70)
    print(" 🏆 同等参数量 A/B 对比终极战报 (SVD 视角)")
    print("="*70)
    print(f" 1. 满血原版 PPL                 :   5.43")
    print(f" 2. SVD Zero-Shot PPL (未微调)  : {ppl_zero:>8.2f}")
    print(f" 3. SVD Healed PPL    (微调后)  : {ppl_healed:>8.2f}")
    print("="*70)

if __name__ == "__main__":
    main()