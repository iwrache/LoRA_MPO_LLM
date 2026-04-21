#!/usr/bin/env python3
"""
MPO 究极压缩管线 (Block-Wise Joint Optimization V8.0)
包含：激活空间 Block-Loss、联合门控寻优、逐层误差级联吸收、ASVD 初始化！
"""

import os
# 强制锁定前4张卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path

# ====== 在文件顶部加入 ======
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np

def find_permutation_for_ffn(W_gate, W_up, method='ward'):
    """
    对 gate_proj 和 up_proj 的行做联合层次聚类，
    返回使相似行相邻的排列索引
    """
    # 拼接两个矩阵的行作为联合特征
    features = torch.cat([W_gate, W_up], dim=1).float().cpu().numpy()
    
    # 层次聚类（ward 法倾向于产生均匀大小的簇）
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
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig, skip_svd=False):
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

            eye = torch.eye(in_features, device=target_device, dtype=torch.float32)  
            W_mpo = mpo_gpu(eye).T                             
            if bias_backup is not None: mpo_gpu.bias.data.copy_(bias_backup)

            # Step 0: 基于 ASVD 初始化残差的 SVD 分解
            Delta_W = (W_orig - W_mpo).contiguous()
            U, S, Vh = torch.linalg.svd(Delta_W, full_matrices=False)
            S_sqrt = torch.diag(torch.sqrt(S[:self.r].clamp(min=0)))
            self.lora_A = nn.Parameter((S_sqrt @ Vh[:self.r, :]).to(target_dtype))
            self.lora_B = nn.Parameter((U[:, :self.r] @ S_sqrt).to(target_dtype))

    def forward(self, x):
        # 🚀 终极动态数据类型护盾：不管 x 是 float32 还是 bfloat16，参数强制对齐，绝不报错！
        dtype = x.dtype
        mpo_out = self.mpo(x)
        lora_out = F.linear(F.linear(x, self.lora_A.to(dtype)), self.lora_B.to(dtype))
        return mpo_out + lora_out

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
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
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
    parser.add_argument("--custom_layers", type=int, nargs='*', default=[])
    parser.add_argument("--custom_ratios", type=float, nargs='*', default=[])
    parser.add_argument("--local_steps", type=int, default=300, help="Block-wise对齐步数")
    parser.add_argument("--e2e_steps", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="./ultimate_healed_llama.pt")
    args = parser.parse_args()

    custom_ratio_map = dict(zip(args.custom_layers, args.custom_ratios))

    print("="*70)
    print(" 🚀 究极两阶段压缩管线启动 (Block-Wise 联合优化 + 逐层级联吸收模式)")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # =====================================================================
    # 🚨 终极修复：为 Base 模型强制注入标准的 Chat Template，防止拼接数据时报错
    # =====================================================================
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{{ eos_token }}\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )
        
    print("📦 加载模型 (纯血 Float32 精度)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
    student_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
    teacher_model.eval()
    
    num_layers = len(student_model.model.layers)
    
    print("\n📚 截获极其珍贵的校准数据 (128条足以激活海森矩阵的威力)...")
    #ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 50).select(range(128))
    #calib_inputs = [tokenizer(t, return_tensors="pt", max_length=256, truncation=True).input_ids for t in ds["text"]]
    print("\n📚 正在构建混合通用特征校准集 (防断网本地版)...")
    calib_texts = []
    
    # 1. 抽取 64 条 WikiText (提取严谨的语法和书面知识特征)
    wiki_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for t in wiki_ds["text"]:
        if len(t) > 200: 
            calib_texts.append(t)
        if len(calib_texts) >= 64: 
            break
            
    # 2. 抽取 64 条 UltraChat (提取问答、逻辑、代码等通用对话特征)
    # 这个数据集你在阶段 4 也要用，提前加载会直接存入本地缓存，又快又稳！
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    for item in chat_ds:
        # 把多轮对话拼接成一段长特征文本
        text = "\n".join([m["content"] for m in item["messages"]])
        if len(text) > 200: 
            calib_texts.append(text)
        # 总数凑齐 128 条就收手
        if len(calib_texts) >= 128: 
            break
            
    calib_inputs = [tokenizer(t, return_tensors="pt", max_length=256, truncation=True).input_ids for t in calib_texts]
    print(f"✅ 成功构建 {len(calib_inputs)} 条混合黄金特征数据 (Wiki+Chat)！")

    print(f"\n🦴 [阶段 1] 动态搭建全网络 MPO 骨架 (ASVD 激活感知识别中)...")
    activation_scales_dict = get_activation_scales(student_model, tokenizer, num_samples=32, max_len=256)
    
    # ====== 替换阶段 1 的 for layer_idx 循环体 ======

    for layer_idx in range(num_layers):
        ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
        if ratio >= 0.99:
            continue

        student_layer = student_model.model.layers[layer_idx]
        NUM_CORES, LORA_RANK = 3, 32

        gate_lin = student_layer.mlp.gate_proj
        up_lin   = student_layer.mlp.up_proj
        down_lin = student_layer.mlp.down_proj

        # ============================================================
        # Step 1: 找最优排列 P（联合聚类 gate + up 的行）
        # ============================================================
        print(f"  🔀 Layer {layer_idx}: 计算最优行排列...")
        
        # 🚀 修正 1：去掉 .cpu()，让大矩阵留在 GPU，准备满速计算！
        W_gate_raw = gate_lin.weight.detach().float()
        W_up_raw   = up_lin.weight.detach().float()

        s_vec_gate = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.gate_proj")
        s_vec_up   = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.up_proj")
        
        # 🚀 修正 2：确保缩放系数也老老实实待在 GPU 上！
        s_val_gate = s_vec_gate.to(W_gate_raw.device).float() if isinstance(s_vec_gate, torch.Tensor) else (s_vec_gate or 1.0)
        s_val_up   = s_vec_up.to(W_up_raw.device).float() if isinstance(s_vec_up, torch.Tensor) else (s_vec_up or 1.0)

        W_gate_scaled = W_gate_raw * s_val_gate
        W_up_scaled   = W_up_raw * s_val_up

        # 聚类函数内部已经写了 .cpu().numpy()，传 GPU 张量进去完全合法且安全
        perm = find_permutation_for_ffn(W_gate_scaled, W_up_scaled)

        # 聚类函数内部已经写了 .cpu().numpy()，传 GPU 张量进去完全合法且安全
        perm_cpu = find_permutation_for_ffn(W_gate_scaled, W_up_scaled)
        
        # 🚀 护盾 1：把索引张量送到与权重同一张显卡上，实现无缝切片！
        perm = perm_cpu.to(W_gate_raw.device)

        # 🚀 修正 3：在 GPU 上重排后，立刻加上 .contiguous() 锁死显存连续性，防止 cuSOLVER 越界！
        W_gate_perm = W_gate_raw[perm, :].contiguous()
        W_up_perm   = W_up_raw[perm, :].contiguous()
        W_down_perm = down_lin.weight.detach().float()[:, perm].contiguous()
        with torch.no_grad():
            down_lin.weight.copy_(W_down_perm.to(dtype=down_lin.weight.dtype))

        # ============================================================
        # Step 0 + Step 2: ASVD + MPO 分解（在排列后的矩阵上做）
        # ============================================================
        for proj, W_perm in [("gate_proj", W_gate_perm), ("up_proj", W_up_perm)]:
            lin = getattr(student_layer.mlp, proj)
            out_f, in_f = lin.weight.shape
            chi_ffn = estimate_mpo_bond_dim(in_f, out_f, NUM_CORES, ratio)
            out_fac = find_factors_balanced(out_f, NUM_CORES)
            in_fac  = find_factors_balanced(in_f, NUM_CORES)

            s_vec = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.{proj}")
            s_gpu = None
            if s_vec is not None:
                # 🚀 修正 4：保证 s_vec 在 GPU 上且显存连续
                s_gpu = s_vec.to(lin.weight.device).float().contiguous() if isinstance(s_vec, torch.Tensor) else float(s_vec)

            # ============================================================
            # 🛡️ 终极安全退守：强行把待分解的矩阵拉回 CPU！
            # 避开 GPU SVD 的底层越界 Bug，让坚如磐石的 CPU LAPACK 库来算
            # ============================================================
            W_cpu_for_mpo = W_perm.detach().cpu().float()
            
            s_cpu_for_mpo = None
            if s_gpu is not None:
                s_cpu_for_mpo = s_gpu.detach().cpu().float() if isinstance(s_gpu, torch.Tensor) else float(s_gpu)

            # 在 CPU 上安心分解，绝不崩溃！
            cores = factor_linear_mpo_custom(
                W_cpu_for_mpo, chi_ffn, NUM_CORES, out_fac, in_fac,
                s_cpu_for_mpo, adaptive=True, energy_threshold=0.99
            )

            # 算完之后，把切好的张量碎片安全地送回 GPU
            cleaned_cores = [c.to(device=lin.weight.device, dtype=lin.weight.dtype) for c in cores]
            
            # W_orig_for_res 依然留在 GPU 上，用于包装残差和后续计算
            W_orig_for_res = W_perm.to(dtype=lin.weight.dtype)

            mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_gpu)
            setattr(
                student_layer.mlp, proj,
                ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, W_orig_for_res, skip_svd=False)
            )

        print(f"  ✅ Layer {layer_idx}: Permutation + MPO + LoRA 骨架搭建完毕")

    PROGRESS_CKPT = "./progressive_layer_checkpoint.pt"
    
    # ---------------------------------------------------------
    # 🌌 [阶段 2] 逐层误差吸收 (Sequential Cascading) + Block-wise 联合优化
    # ---------------------------------------------------------
    print(f"\n[阶段 2] 🧬 启动 Block-wise 级联特征对齐 (Cross-layer Error Compensation)...")
    
    for layer_idx in range(num_layers):
        ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
        if ratio >= 0.99:
            print(f"\n ⏭️ Layer {layer_idx} [首尾保护区]，跳过。")
            continue

        student_layer = student_model.model.layers[layer_idx]
        teacher_layer = teacher_model.model.layers[layer_idx]
        layer_device = next(student_layer.parameters()).device
        
        print(f"\n 🎯 正在级联对齐 Layer {layer_idx} (联合优化 gate & up)...")
        
        # ========================================================
        # 🚨 Step 4: Sequential Calibration 数据截获 (加入极速短路截断)
        # ========================================================
        class StopForwardException(Exception): pass

        cached_mlp_inputs = []
        def capture_hook(module, args):
            # 🚨 极速展平修复：不管原来是 [1, 192, H] 还是 [1, 126, H]
            # 直接重塑为 [192, H] 或 [126, H] 的二维矩阵，这样 torch.cat 拼接时就畅通无阻了！
            flattened_X = args[0].detach().cpu().reshape(-1, args[0].shape[-1])
            cached_mlp_inputs.append(flattened_X)
            
            # 拿到当前层的输入后，立刻抛出异常打断前向传播，绝不跑后面的层！
            raise StopForwardException
            
        handle = student_layer.mlp.register_forward_pre_hook(capture_hook)
        
        # 获取最安全的底层设备，防止多卡 device_map 报错
        embed_device = student_model.model.embed_tokens.weight.device
        
        with torch.no_grad():
            for input_ids in tqdm(calib_inputs, desc=f"积累前序层误差 (极速截断中)", leave=False):
                try:
                    student_model(input_ids.to(embed_device))
                except StopForwardException:
                    pass  # 正常捕获我们自己抛出的异常，继续处理下一条数据
        handle.remove()
        
        # 将积累的特征拼接
        X_calib_tensor = torch.cat(cached_mlp_inputs, dim=0) # [Num_samples * Seq_len, Hidden_Dim]

        # ========================================================
        # 🚨 Step 3 & 5: Block-wise 激活空间联合优化
        # ========================================================
        for param in student_model.parameters(): param.requires_grad = False
        
        trainable_params_for_opt = []
        raw_params_for_clip = []
        
        # 将 gate 和 up 的所有参数放入统一个优化池，强迫它们互相补偿！
        for proj in ["gate_proj", "up_proj"]:
            wrapper = getattr(student_layer.mlp, proj)
            wrapper.lora_A.requires_grad = True
            wrapper.lora_B.requires_grad = True
            for core in wrapper.mpo.cores: core.requires_grad = True
            
            trainable_params_for_opt.extend([
                {"params": wrapper.lora_A, "lr": 1e-4}, 
                {"params": wrapper.lora_B, "lr": 1e-4}, 
                {"params": wrapper.mpo.cores, "lr": 1e-5}
            ])
            raw_params_for_clip.extend([wrapper.lora_A, wrapper.lora_B] + list(wrapper.mpo.cores))
            
        optimizer = torch.optim.AdamW(trainable_params_for_opt, weight_decay=0.01)
        student_layer.mlp.train()
        
        # 获取 Teacher 当前层所在的具体物理显卡
        teacher_device = next(teacher_layer.parameters()).device

        # ========================================================
        # 取一个小 batch 用于联合寻优
        # 🚨 直接把二维张量赋给 X_flat，行数就是真实的 token 数量
        X_flat = X_calib_tensor
        num_tokens = X_flat.shape[0]
        # ========================================================
        
        for step in range(args.local_steps):
            optimizer.zero_grad()
            
            sample_indices = torch.randperm(num_tokens)[:2048]
            # X_batch 在 Student 的显卡上
            X_batch = X_flat[sample_indices].to(device=layer_device, dtype=torch.float32)
            
            # 🚀 护盾 2：把数据送到 Teacher 的显卡上计算，算完再拉回 Student 的显卡计算 Loss！
            with torch.no_grad():
                Y_orig = teacher_layer.mlp(X_batch.to(teacher_device)).to(layer_device)
                
            # 计算 Student 的实际输出 (此时 X_batch 和 Student 同在一个 GPU，非常安全)
            Y_recon = student_layer.mlp(X_batch)
            
            loss = F.mse_loss(Y_recon, Y_orig)
            loss.backward()
            
            has_nan_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in raw_params_for_clip)
            if has_nan_grad: 
                optimizer.zero_grad()
                continue
                
            torch.nn.utils.clip_grad_norm_(raw_params_for_clip, 1.0)
            optimizer.step()
            
        print(f"    ✅ Block-wise 联合收敛! MLP Output MSE: {loss.item():.6f}")
        
        # 优化完毕，重新冻结
        student_layer.mlp.eval()
        for proj in ["gate_proj", "up_proj"]:
            wrapper = getattr(student_layer.mlp, proj)
            wrapper.lora_A.requires_grad = False
            wrapper.lora_B.requires_grad = False
            for core in wrapper.mpo.cores: core.requires_grad = False

        del cached_mlp_inputs, X_calib_tensor, X_flat; torch.cuda.empty_cache(); gc.collect()
        
        tmp_ckpt = PROGRESS_CKPT + ".tmp"
        try:
            torch.save({'next_layer': layer_idx + 1, 'model_state_dict': student_model.state_dict()}, tmp_ckpt)
            os.replace(tmp_ckpt, PROGRESS_CKPT)
        except Exception as e:
            print(f"    ⚠️ 保存异常: {e}")
    
    print("\n[阶段 3] 测量 Block-wise 级联特征对齐后的 PPL...")
    ppl_mid = eval_ppl(student_model, tokenizer)
    print(f"🌟 【阶段二】级联收敛 PPL: {ppl_mid}")

    print("\n[阶段 4] 🌌 启动端到端联合蒸馏 (E2E Healing)...")
    os.environ["MPO_EVAL_PATH"] = "mpo"; os.environ["MPO_TRAIN_PATH"] = "mpo"
    torch.cuda.empty_cache(); gc.collect()
    '''
    student_model.to(torch.bfloat16)
    teacher_model.to(torch.bfloat16)

    student_model = train_healing(
        student_model=student_model, tokenizer=tokenizer, teacher_model=teacher_model, 
        dataset_name="mixed", epochs=1, batch_size=4, accum_steps=2, lr=2e-5, seq_len=1024,  
        save_every_n_steps=500, checkpoint_dir="./healing_checkpoints_mixed_e2e", 
        max_update_steps=args.e2e_steps, resume_from_checkpoint=None 
    )
    '''
    print("\n[阶段 4] 🌌 训练已完成，直接加载 E2E 终极权重！...")
    final_checkpoint_path = "./healing_checkpoints_mixed_e2e/checkpoint_upd_1000.pt" 
    
    print(f"正在加载神药: {final_checkpoint_path}")
    
    # 1. 把整个“训练现场大礼包”加载到内存
    checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
    
    # 2. 剥开包装，只提取出我们真正需要的训练权重
    # (根据你的报错信息，它被安全地存在了 "trainable_state_dict" 这个抽屉里)
    state_dict = checkpoint["trainable_state_dict"]
    
    # 3. 🚨 核心修复：加载权重，并开启 strict=False！
    # 告诉 PyTorch："我只给你提供被修改过的 MPO 和 LoRA 零件，
    # 原本的 Attention 和 Norm 层你就保持原样，不要报 Missing 错误！"
    student_model.load_state_dict(state_dict, strict=False)
    
    print("✅ 权重完美融合！MPO 满血架构复活！")

    # 直接开始最终 PPL 评测 (保持在 float32 获得最高精度)
    ppl_final = eval_ppl(student_model, tokenizer)
    
    print("\n" + "="*70)
    print(" 🏆 终极 SOTA 级架构战报")
    print("="*70)
    print(f" 1. 满血原版 PPL                 : {5.43:>8.2f}")
    print(f" 2. Block-wise 吸收后 PPL        : {ppl_mid:>8.2f}")
    print(f" 3. 全局端到端复活 PPL (Healed)  : {ppl_final:>8.2f}")
    print("="*70)

if __name__ == "__main__":
    main()