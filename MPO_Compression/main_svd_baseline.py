#!/usr/bin/env python3
"""
SVD 究极压缩管线 (Block-Wise Joint Optimization V8.0)
包含：激活空间 Block-Loss、联合门控寻优、逐层误差级联吸收、ASVD 初始化！
"""

import os
os.environ["HF_HOME"] = "/mnt/hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hf-cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from accelerate import Accelerator

from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np

def find_permutation_for_ffn(W_gate, W_up, method='ward'):
    """对 gate_proj 和 up_proj 的行做联合层次聚类，返回使相似行相邻的排列索引"""
    features = torch.cat([W_gate, W_up], dim=1).float().cpu().numpy()
    Z = linkage(features, method=method, metric='euclidean')
    perm = leaves_list(Z)
    return torch.tensor(perm, dtype=torch.long)

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales
from healing import train_healing

class SVDLinear(nn.Module):
    """
    同等参数量下的 SVD 骨架 (内置 ASVD 激活缩放逻辑，保证与 MPO 对决的公平性)
    """
    def __init__(self, in_features, out_features, rank, W_orig, s_vector=None, skip_init=False):
        super().__init__()
        target_dtype, target_device = W_orig.dtype, W_orig.device
        
        # SVD 分解出两个小矩阵: B (rank x in_features) 和 A (out_features x rank)
        self.B = nn.Linear(in_features, rank, bias=False, device=target_device, dtype=target_dtype)
        self.A = nn.Linear(rank, out_features, bias=False, device=target_device, dtype=target_dtype)
        self.bias = None # 保持与 MPO 接口一致
        
        if skip_init:
            return
            
        with torch.no_grad():
            W_float = W_orig.float()
            
            # 1. 激活值缩放 (ASVD 逻辑)
            if s_vector is not None:
                s_vec_safe = torch.clamp(s_vector.to(target_device).float(), min=1e-6)
                W_scaled = W_float * s_vec_safe.unsqueeze(0)
            else:
                W_scaled = W_float
            
            # 2. 安全 SVD 分解
            try:
                U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
            except RuntimeError:
                print("      [⚠️ GPU SVD 内存碎片, 降级至 CPU 计算 Vanilla SVD...]")
                U, S, Vh = torch.linalg.svd(W_scaled.cpu(), full_matrices=False)
                U, S, Vh = U.to(target_device), S.to(target_device), Vh.to(target_device)
            
            # 3. 截断到目标 Rank 并平分奇异值
            S_sqrt = torch.diag(torch.sqrt(S[:rank].clamp(min=0)))
            A_weight = U[:, :rank] @ S_sqrt
            B_weight = S_sqrt @ Vh[:rank, :]
            
            # 4. 反向缩放还原
            if s_vector is not None:
                # 对应于除以 s_vector
                B_weight = B_weight / s_vec_safe.unsqueeze(0)
            
            # 5. 注入参数
            self.B.weight.data.copy_(B_weight.to(target_dtype))
            self.A.weight.data.copy_(A_weight.to(target_dtype))

    def forward(self, x): 
        # 严格执行 B -> A 的两步线性投影
        return self.A(self.B(x))

        
def compute_mpo_core_shapes(out_fac, in_fac, bond_dim, num_cores):
    shapes, prev = [], 1
    for k in range(num_cores - 1):
        rows = prev * out_fac[k] * in_fac[k]
        cols = 1
        for j in range(k + 1, num_cores): cols *= out_fac[j] * in_fac[j]
        r = max(1, min(bond_dim, min(rows, cols)))
        shapes.append((prev, out_fac[k], in_fac[k], r))
        prev = r
    shapes.append((prev, out_fac[-1], in_fac[-1], 1))
    return shapes

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
                # 🌟 修正 3：确保 s_vec 是张量类型，防止 float 报错
                s_vec = torch.tensor(s_vector, device=target_device, dtype=torch.float32) if not isinstance(s_vector, torch.Tensor) else s_vector.to(target_device).float()
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

# 🌟 修正 1：完美修复 PPL 数学计算逻辑
def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=15000):
    model.eval() 
    print("⏳ 正在计算 Wikitext-2 PPL...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    
    # 获取正确的设备位置
    device = model.model.embed_tokens.weight.device
    input_ids = enc.input_ids.to(device)
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    effective_max = min(max_len, max_ctx)
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    total_loss = 0.0
    total_tokens = 0

    for begin in tqdm(range(0, seq_len, stride), desc="PPL 评测中"):
        end = min(begin + effective_max, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1: continue
        with torch.no_grad():
            out = model(chunk)
            lm_logits = out.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            
            # 使用 sum 而不是 mean，累加所有 token 的 loss
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                total_tokens += shift_labels.numel()

    if total_tokens == 0: return float("nan")
    return round(torch.exp(torch.tensor(total_loss / total_tokens)).item(), 2)

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
    parser.add_argument("--custom_layers", type=int, nargs='*', default=[])
    parser.add_argument("--custom_ratios", type=float, nargs='*', default=[])
    parser.add_argument("--local_steps", type=int, default=300)
    parser.add_argument("--e2e_steps", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="/mnt/sx_data/ultimate_healed_llama.pt")
    
    parser.add_argument("--adaptive_mode", type=str, default="entropy", choices=["fixed", "energy", "entropy", "quantum"])
    parser.add_argument("--quantum_scale", type=float, default=0.6)
    parser.add_argument("--disable_asvd", action="store_true")
    parser.add_argument("--disable_perm", action="store_true")
    args = parser.parse_args()

    custom_ratio_map = dict(zip(args.custom_layers, args.custom_ratios))
    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")
    device = accelerator.device  
    
    print("="*70)
    print(" 🚀 究极两阶段压缩管线启动 (Block-Wise 联合优化 + 逐层级联吸收模式)")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{{ eos_token }}\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )
        
    print("📦 加载模型...")

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)         

    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)         
    teacher_model.eval()
    
    num_layers = len(student_model.model.layers)
    
    print("\n📚 正在构建混合通用特征校准集 (防断网本地版)...")
    calib_texts = []
    
    wiki_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for t in wiki_ds["text"]:
        if len(t) > 200: calib_texts.append(t)
        if len(calib_texts) >= 64: break
            
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    for item in chat_ds:
        text = "\n".join([m["content"] for m in item["messages"]])
        if len(text) > 200: calib_texts.append(text)
        if len(calib_texts) >= 128: break
            
    calib_inputs = [tokenizer(t, return_tensors="pt", max_length=256, truncation=True).input_ids for t in calib_texts]
    print(f"✅ 成功构建 {len(calib_inputs)} 条混合黄金特征数据 (Wiki+Chat)！")

    PROGRESS_CKPT = "/mnt/sx_data/progressive_layer_checkpoint.pt"
    
    skip_to_phase_4 = False
    ppl_mid = 0.0  # 🌟 修复变量未定义崩溃
    
    if os.path.exists(PROGRESS_CKPT):
        if accelerator.is_main_process:
            print(f"\n📦 检测到阶段 2 的完美存档：{PROGRESS_CKPT}")
            print("🚀 直接跳过前三个阶段，启动『伪骨架搭建』与『权重注入』...")
        skip_to_phase_4 = True 

    if skip_to_phase_4:
        if accelerator.is_main_process: print("骨架重塑中...")
        ckpt = torch.load(PROGRESS_CKPT, map_location="cpu")
        state_dict = ckpt['model_state_dict']
        
        for layer_idx in range(student_model.config.num_hidden_layers):
            student_layer = student_model.model.layers[layer_idx]
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                lin = getattr(student_layer.mlp, proj_name)
                in_f, out_f = lin.weight.shape[1], lin.weight.shape[0]
                
                # ==== 替换为兼容双模式的重构逻辑 ====
                mpo_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.mpo.cores."
                svd_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.mpo.A.weight"
                
                if svd_prefix in state_dict:
                    # 如果存档里有 A.weight，说明是 SVD 的存档
                    rank = state_dict[svd_prefix].shape[1]
                    mpo = SVDLinear(in_f, out_f, rank, lin.weight, skip_init=True)
                else:
                    # 否则正常重构 MPO
                    core_keys = [k for k in state_dict.keys() if k.startswith(mpo_prefix)]
                    NUM_CORES = len(core_keys)
                    if NUM_CORES == 0: continue 
                    real_cores = []
                    for i in range(NUM_CORES):
                        real_cores.append(torch.nn.Parameter(state_dict[f"{mpo_prefix}{i}"].clone().to(device=lin.weight.device)))
                    mpo = MPOLinear(in_f, out_f, real_cores)
                
                # 动态获取 LoRA Rank (保持不变)
                lora_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.lora_A"
                LORA_RANK = state_dict[lora_prefix].shape[0] if lora_prefix in state_dict else 32
                
                setattr(student_layer.mlp, proj_name, ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight, skip_svd=True))
                
        student_model.load_state_dict(state_dict, strict=False)
        del ckpt, state_dict; gc.collect(); torch.cuda.empty_cache()
    else:        
        print(f"\n🦴 [阶段 1] 动态搭建全网络 MPO 骨架...")
        if args.disable_asvd:
            activation_scales_dict = {} 
        else:
            activation_scales_dict = get_activation_scales(student_model, tokenizer, num_samples=32, max_len=256)
        
        for layer_idx in range(num_layers):
            ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
            if ratio >= 0.99: continue

            student_layer = student_model.model.layers[layer_idx]
            NUM_CORES, LORA_RANK = 3, 32
            gate_lin = student_layer.mlp.gate_proj
            up_lin   = student_layer.mlp.up_proj
            down_lin = student_layer.mlp.down_proj

            W_gate_raw = gate_lin.weight.detach().float()
            W_up_raw   = up_lin.weight.detach().float()

            s_vec_gate = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.gate_proj")
            s_vec_up   = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.up_proj")
            
            # 🌟 修正 2：防止 Python float 与 Tensor 交互低效
            s_val_gate = s_vec_gate.to(W_gate_raw.device).float() if isinstance(s_vec_gate, torch.Tensor) else torch.tensor(1.0, device=W_gate_raw.device)
            s_val_up   = s_vec_up.to(W_up_raw.device).float() if isinstance(s_vec_up, torch.Tensor) else torch.tensor(1.0, device=W_up_raw.device)
            W_gate_scaled = W_gate_raw * s_val_gate
            W_up_scaled   = W_up_raw * s_val_up

            if getattr(args, 'disable_perm', False):
                W_gate_perm = W_gate_raw.contiguous()
                W_up_perm   = W_up_raw.contiguous()
                W_down_perm = down_lin.weight.detach().float().contiguous()
                intermediate_size = W_gate_raw.shape[0]
                perm = torch.arange(intermediate_size, device=W_gate_raw.device)
            else:
                perm_cpu = find_permutation_for_ffn(W_gate_scaled, W_up_scaled)
                perm = perm_cpu.to(W_gate_raw.device)

                W_gate_perm = W_gate_raw[perm, :].contiguous()
                W_up_perm   = W_up_raw[perm, :].contiguous()
                W_down_perm = down_lin.weight.detach().float()[:, perm].contiguous()
                
                # 🌟 修正 4：保证 3 个矩阵权重完整写回
                with torch.no_grad():
                    down_lin.weight.copy_(W_down_perm.to(dtype=down_lin.weight.dtype))
                    gate_lin.weight.copy_(W_gate_perm.to(dtype=gate_lin.weight.dtype))
                    up_lin.weight.copy_(W_up_perm.to(dtype=up_lin.weight.dtype))

            for proj, W_perm, is_down in [
                ("gate_proj", W_gate_perm, False), 
                ("up_proj", W_up_perm, False), 
                ("down_proj", W_down_perm, True)
            ]:
                lin = getattr(student_layer.mlp, proj)
                out_f, in_f = lin.weight.shape
                # ================= 替换为：严格同等参数量的 SVD 截断 =================
                orig_params = out_f * in_f
                target_params = int(orig_params * ratio)      # 总预算
                lora_params = LORA_RANK * (in_f + out_f)      # LoRA 固定开销
                svd_budget = target_params - lora_params      # 留给 SVD 的预算
                
                # 核心数学：SVD 参数量 = rank * in_f + rank * out_f
                svd_rank = svd_budget // (in_f + out_f)
                svd_rank = max(1, min(svd_rank, min(in_f, out_f))) # 安全边界
                
                # 实例化我们刚才写的 SVDLinear (它现在伪装成了 MPO 对象)
                mpo = SVDLinear(in_f, out_f, svd_rank, W_perm, s_vector=s_gpu)
                
                actual_svd_params = svd_rank * (in_f + out_f)
                actual_total_params = actual_svd_params + lora_params
                actual_ratio = actual_total_params / orig_params
                
                if accelerator.is_main_process:
                    print(f"      ↳ [{proj}] 同等预算 SVD 截断 -> Rank: {svd_rank} | 压缩率: {actual_ratio*100:.2f}% (SVD: {actual_svd_params}, LoRA: {lora_params})")
                
                # 接回你原来的 ResMPOWrapper (它不知道内部换成了 SVD，它依然能完美工作！)
                W_orig_for_res = W_perm.to(dtype=lin.weight.dtype)
                setattr(student_layer.mlp, proj, ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, W_orig_for_res, skip_svd=False, s_vector=s_gpu))

        if accelerator.is_main_process:
            orig_total_params = sum(p.numel() for p in teacher_model.parameters())
            new_total_params = sum(p.numel() for p in student_model.parameters())
            print("\n" + "="*70)
            print(" 📊 [阶段 1 总结] 量子自组织压缩战报")
            print(f" 原始模型总参数量 : {orig_total_params / 1e9:.4f} B (Billion)")
            print(f" 压缩后模型总参数量: {new_total_params / 1e9:.4f} B (Billion)")
            print(f" 🌟 全模型真实保留率: {(new_total_params / orig_total_params) * 100:.2f}%")
            print("="*70)

        # ---------------------------------------------------------
        # 🌌 [阶段 2] 逐层误差吸收 (Sequential Cascading) 
        # ---------------------------------------------------------
        print(f"\n[阶段 2] 🧬 启动 Block-wise 级联特征对齐 (多卡冗余同步计算中)...")
        
        for layer_idx in range(num_layers):
            ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
            if ratio >= 0.99: continue

            student_layer = student_model.model.layers[layer_idx]
            teacher_layer = teacher_model.model.layers[layer_idx]
            layer_device = next(student_layer.parameters()).device
            
            if accelerator.is_main_process: print(f"\n 🎯 正在级联对齐 Layer {layer_idx} (仅优化 LoRA)...")
            
            class StopForwardException(Exception): pass
            cached_mlp_inputs = []
            
            def capture_hook(module, args):
                flattened_X = args[0].detach().cpu().reshape(-1, args[0].shape[-1])
                cached_mlp_inputs.append(flattened_X)
                raise StopForwardException
                
            handle = student_layer.mlp.register_forward_pre_hook(capture_hook)
            embed_device = student_model.model.embed_tokens.weight.device
            
            with torch.no_grad():
                for input_ids in tqdm(calib_inputs, desc=f"积累前序层误差", leave=False, disable=not accelerator.is_main_process):
                    try: student_model(input_ids.to(embed_device))
                    except StopForwardException: pass 
            handle.remove()
            X_calib_tensor = torch.cat(cached_mlp_inputs, dim=0)

            for param in student_model.parameters(): param.requires_grad = False
            
            trainable_params_for_opt = []
            raw_params_for_clip = []
            
            # 🌟 核心修复：绝对冻结 MPO，只练 LoRA！
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wrapper = getattr(student_layer.mlp, proj)
                wrapper.lora_A.requires_grad = True
                wrapper.lora_B.requires_grad = True
                if hasattr(wrapper.mpo, 'cores'):
                    for core in wrapper.mpo.cores: core.requires_grad = False
                else: # SVD 模式兼容
                    wrapper.mpo.A.weight.requires_grad = False
                    wrapper.mpo.B.weight.requires_grad = False
                
                trainable_params_for_opt.extend([
                    {"params": wrapper.lora_A, "lr": 1e-4}, 
                    {"params": wrapper.lora_B, "lr": 1e-4}
                ])
                raw_params_for_clip.extend([wrapper.lora_A, wrapper.lora_B])
                
            optimizer = torch.optim.AdamW(trainable_params_for_opt, weight_decay=0.01)
            student_layer.mlp.train()
            teacher_device = next(teacher_layer.parameters()).device

            X_flat = X_calib_tensor
            num_tokens = X_flat.shape[0]
            
            for step in range(args.local_steps):
                optimizer.zero_grad()
                
                # 🌟 修复：多卡并行撕裂护盾，确保三卡采样的随机数绝对一致！
                torch.manual_seed(42 + layer_idx * 10000 + step)
                sample_indices = torch.randperm(num_tokens)[:2048]
                
                current_dtype = next(student_layer.parameters()).dtype
                X_batch = X_flat[sample_indices].to(device=layer_device, dtype=current_dtype)
                
                # 🌟 修正 5：加上混合精度护航，并将 backward 移出 autocast 块！
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        Y_orig = teacher_layer.mlp(X_batch.to(teacher_device)).to(layer_device)
                    Y_recon = student_layer.mlp(X_batch)
                    loss = F.mse_loss(Y_recon, Y_orig)
                
                loss.backward()
                
                has_nan_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in raw_params_for_clip)
                if has_nan_grad: 
                    optimizer.zero_grad()
                    continue
                    
                torch.nn.utils.clip_grad_norm_(raw_params_for_clip, 1.0)
                optimizer.step()
                
            if accelerator.is_main_process: print(f"    ✅ Block-wise 联合收敛! MLP MSE: {loss.item():.6f}")
            
            student_layer.mlp.eval()
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                wrapper = getattr(student_layer.mlp, proj)
                wrapper.lora_A.requires_grad = False
                wrapper.lora_B.requires_grad = False

            del optimizer, cached_mlp_inputs, X_calib_tensor, X_flat; torch.cuda.empty_cache(); gc.collect()
            accelerator.wait_for_everyone() 

            # ========================================================
            # 阶段 2 结尾的保存逻辑
            # ========================================================
            if accelerator.is_main_process:
                try:
                    tmp_ckpt = PROGRESS_CKPT + f".tmp_{layer_idx}"
                    print(f"    💾 正在主卡提取权重...")
                    state_dict = accelerator.unwrap_model(student_model).state_dict()
                    
                    print(f"    💾 正在写入临时文件...")
                    torch.save({'next_layer': layer_idx + 1, 'model_state_dict': state_dict}, tmp_ckpt)
                    
                    print(f"    💾 正在替换正式文件...")
                    os.replace(tmp_ckpt, PROGRESS_CKPT)
                    print(f"    ✅ Layer {layer_idx} 级联存档成功落盘！")
                except Exception as e:
                    print(f"    🚨 致命警告：保存异常！{e}")
            
            # 第一道门：所有 GPU 在这里等 GPU 0 把文件写完
            accelerator.wait_for_everyone()
            
            # 🌟 核心修复：多卡浮点数漂移强制对齐（所有 GPU 共同执行）
            if os.path.exists(PROGRESS_CKPT):
                if accelerator.is_main_process:
                    print(f"    🔄 [多卡同步] 主卡已落盘，正在强制所有 GPU 对齐权重...")
                
                ckpt_sync = torch.load(PROGRESS_CKPT, map_location='cpu')
                student_model.load_state_dict(ckpt_sync['model_state_dict'], strict=False)
                del ckpt_sync
                gc.collect()
                torch.cuda.empty_cache()
            
            # 第二道门：所有 GPU 等待大家全都加载完毕，再一起整齐划一地走向下一层
            accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            print("\n[阶段 3] 测量 Block-wise 级联特征对齐后的 PPL...")
            ppl_mid = eval_ppl(student_model, tokenizer)
            print(f"🌟 【阶段二】级联收敛 PPL: {ppl_mid}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process: print("\n[阶段 4] 🌌 启动端到端联合蒸馏 (E2E Healing)...")
    os.environ["MPO_EVAL_PATH"] = "mpo"; os.environ["MPO_TRAIN_PATH"] = "mpo"
    torch.cuda.empty_cache(); gc.collect()
    
    student_model.to(torch.bfloat16)
    teacher_model.to(torch.bfloat16)
    
    exp_suffix = "full"
    if getattr(args, 'disable_asvd', False): exp_suffix += "_noASVD"
    if getattr(args, 'disable_perm', False): exp_suffix += "_noPerm"
    
    HEALING_DIR = f"/mnt/sx_data/healing_checkpoints_mixed_{exp_suffix}"
    FINAL_CKPT = f"{HEALING_DIR}/checkpoint_upd_1000.pt" 
    
    if os.path.exists(FINAL_CKPT):
        if accelerator.is_main_process:
            print(f"\n📦 检测到已完成的 [{exp_suffix}] 实验存档：{FINAL_CKPT}")
        try:
            checkpoint = torch.load(FINAL_CKPT, map_location="cpu")
            state_dict = checkpoint.get("trainable_state_dict", checkpoint)
            student_model.load_state_dict(state_dict, strict=False)
            del checkpoint, state_dict; gc.collect()
        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ 警告：检测到损坏的存档文件 ({e})，将重新开始训练！")
                os.remove(FINAL_CKPT)
            accelerator.wait_for_everyone()
            
    if not os.path.exists(FINAL_CKPT):
        student_model = train_healing(
            student_model=student_model, tokenizer=tokenizer, teacher_model=teacher_model, 
            dataset_name="mixed", epochs=1, batch_size=1, accum_steps=16, lr=2e-5, seq_len=1024,  
            save_every_n_steps=500, checkpoint_dir=HEALING_DIR, 
            max_update_steps=args.e2e_steps, resume_from_checkpoint=None, accelerator=accelerator
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n🏆 开始计算压缩+康复微调后的最终 PPL...")
        student_model.eval()
        student_model.to(torch.bfloat16)
        final_ppl = eval_ppl(student_model, tokenizer)
        
        print("\n" + "="*70)
        print(" 🏆 终极 SOTA 级架构战报")
        print("="*70)
        print(f" 1. 满血原版 PPL                 : {5.43:>8.2f}")
        print(f" 2. Block-wise 吸收后 PPL        : {ppl_mid:>8.2f}")
        print(f" 3. 全局端到端复活 PPL (Healed)  : {final_ppl:>8.2f}")
        print("="*70)

if __name__ == "__main__":
    main()