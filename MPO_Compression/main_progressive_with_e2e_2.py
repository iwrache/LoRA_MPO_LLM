#!/usr/bin/env python3
"""
MPO 究极两阶段压缩管线 (Platinum V3.0)
包含：全精度 Float32、对称U型保护、Teacher-Forcing、参数雷达、原子断点续传！
"""

import os
import sys
import gc
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb
from pathlib import Path

# 强制使用 4 张卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales
from healing import train_healing


import warnings

warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing"
)

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
            mpo_cpu = self.mpo.cpu().float()
            if hasattr(mpo_cpu, 's_vector') and mpo_cpu.s_vector is not None:
                mpo_cpu.s_vector = mpo_cpu.s_vector.cpu().float()
            bias_backup = mpo_cpu.bias.data.clone() if hasattr(mpo_cpu, 'bias') and mpo_cpu.bias is not None else None
            if bias_backup is not None: mpo_cpu.bias.data.zero_()

            eye = torch.eye(in_features, dtype=torch.float32)  
            W_mpo = mpo_cpu(eye).T                             
            if bias_backup is not None: mpo_cpu.bias.data.copy_(bias_backup)

            Delta_W = W_orig.cpu().float() - W_mpo
            U, S, Vh = torch.linalg.svd(Delta_W, full_matrices=False)
            S_sqrt = torch.diag(torch.sqrt(S[:self.r].clamp(min=0)))
            self.lora_A = nn.Parameter((S_sqrt @ Vh[:self.r, :]).to(target_dtype).to(target_device))
            self.lora_B = nn.Parameter((U[:, :self.r] @ S_sqrt).to(target_dtype).to(target_device))
            
            del eye, W_mpo, Delta_W, U, S, Vh

        self.mpo = self.mpo.to(device=target_device, dtype=target_dtype)
        if hasattr(self.mpo, 's_vector') and self.mpo.s_vector is not None:
            self.mpo.s_vector = self.mpo.s_vector.to(device=target_device, dtype=target_dtype)

    def forward(self, x):
        return self.mpo(x) + F.linear(F.linear(x, self.lora_A), self.lora_B)

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
    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return round(ppl, 2)

def print_param_compression_report(teacher_model, student_model):
    orig_params = sum(p.numel() for p in teacher_model.parameters())
    new_params = sum(p.numel() for p in student_model.parameters())
    orig_mlp_params = sum(p.numel() for name, p in teacher_model.named_parameters() if "mlp" in name)
    new_mlp_params = sum(p.numel() for name, p in student_model.named_parameters() if "mlp" in name)

    print("\n" + "="*70)
    print(" 📊 模型参数宏观压缩战报 (全局视野)")
    print("="*70)
    print(f" 🎯 【FC 层 (MLP Block) 专属统计】")
    print(f"    原 Dense 参数: {orig_mlp_params / 1e6:>8.2f} M")
    print(f"    现 混合 参数:  {new_mlp_params / 1e6:>8.2f} M (含 MPO + LoRA + 未压缩的层)")
    print(f"    🌟 MLP 压缩率: {new_mlp_params / orig_mlp_params:>8.2%}")
    print(f"\n 🌐 【全局总计 (包含 Attention 与词表)】")
    print(f"    原模型总参数:  {orig_params / 1e6:>8.2f} M")
    print(f"    现模型总参数:  {new_params / 1e6:>8.2f} M")
    print(f"    📉 绝对下降量: {(orig_params - new_params) / 1e6:>8.2f} M 个参数已被物理切除！")
    print(f"    🌟 全局保留率: {new_params / orig_params:>8.2%}")
    print("="*70)

# =======================================================
# 🚀 全新升级：首尾绝对保护的对称 U 型保留率公式
# =======================================================
def get_u_shape_ratio(layer_idx, total_layers, target_ratio):
    # 首尾各 3 层完全不压，保留率 1.0 (100%)
    if layer_idx < 3 or layer_idx >= total_layers - 3:
        return 1.0
        
    # 中间部分构成完美的对称 U 型抛物线
    start_idx = 3
    end_idx = total_layers - 4
    center = (start_idx + end_idx) / 2.0
    half_range = (end_idx - start_idx) / 2.0
    
    # 归一化到 [-1, 1]
    x = (layer_idx - center) / half_range
    
    # 抛物线方程: 最低点由 target_ratio 决定，两端翘起
    base = max(0.1, target_ratio - 0.15)
    amplitude = 0.45 
    
    ratio = base + amplitude * (x ** 2)
    # 将 U 型部分的最大保留率限制在 0.95，防止和首尾 100% 混淆
    return max(0.1, min(0.95, ratio))

def main():
    parser = argparse.ArgumentParser(description="MPO 究极控制台")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--target_ratio", type=float, default=0.6, help="全局U型基准保留率")
    parser.add_argument("--custom_layers", type=int, nargs='*', default=[], help="需自定义保留率的层号")
    parser.add_argument("--custom_ratios", type=float, nargs='*', default=[], help="对应的保留率")
    parser.add_argument("--local_steps", type=int, default=200, help="局部累进缝合步数")
    parser.add_argument("--e2e_steps", type=int, default=1000, help="E2E 端到端步数")
    parser.add_argument("--save_path", type=str, default="./ultimate_healed_llama.pt")
    args = parser.parse_args()

    custom_ratio_map = dict(zip(args.custom_layers, args.custom_ratios))

    print("="*70)
    print(" 🚀 究极两阶段压缩管线启动 (Float32 满血精度 + 首尾保护对称 U 型)")
    if custom_ratio_map: print(f" 🎯 注入了自定义层压缩率: {custom_ratio_map}")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}</s>\n"
            "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}</s>\n"
            "{% elif message['role'] == 'system' %}<|system|>\n{{ message['content'] }}</s>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )
        
    print("📦 加载模型 (使用纯血 Float32 精度防 NaN)...")
    # 🚨 已经为你全部替换为 Float32！
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
    student_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
    teacher_model.eval()

    print("\n[阶段 0/4] 测量原始满血模型的 PPL (作为无损标杆)...")
    ppl_orig = eval_ppl(teacher_model, tokenizer)
    print(f"🌟 【原始标杆】满血 PPL: {ppl_orig}")

    PROGRESS_CKPT = "./progressive_layer_checkpoint.pt"
    num_layers = len(student_model.model.layers)
    # 因为首尾不压，我们从 Layer 0 开始扫描
    start_layer = 0 
    is_resume = False
    ckpt = None
    
    if os.path.exists(PROGRESS_CKPT):
        print(f"\n📂 检测到中断的手术存档！读取进度坐标...")
        ckpt = torch.load(PROGRESS_CKPT, map_location="cpu")
        start_layer = ckpt.get('next_layer', 0)
        is_resume = True

    print(f"\n[阶段 1/4] 🦴 正在动态搭建全网络 MPO 骨架 (遵循对称 U 型保护策略)...")
    activation_scales_dict = get_activation_scales(student_model, tokenizer, num_samples=32, max_len=256)
    
    # 从 0 到 31 层全盘扫描
    for layer_idx in range(num_layers):
        ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
        
        # 🚨 如果保留率是 100% (首尾层)，直接跳过不拆！
        if ratio >= 0.99:
            print(f"    🛡️ Layer {layer_idx} 属于首尾保护区，跳过 MPO 压缩！")
            continue
            
        student_layer = student_model.model.layers[layer_idx]
        NUM_CORES, LORA_RANK = 3, 32
        
        for proj in ["gate_proj", "up_proj"]:
            lin = getattr(student_layer.mlp, proj)
            out_f, in_f = lin.weight.shape
            chi_ffn = estimate_mpo_bond_dim(in_f, out_f, NUM_CORES, ratio)
            out_fac, in_fac = find_factors_balanced(out_f, NUM_CORES), find_factors_balanced(in_f, NUM_CORES)
            s_vec = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.{proj}")

            if is_resume and layer_idx < start_layer:
                core_shapes = []
                for c_idx in range(NUM_CORES):
                    weight_name = f"model.layers.{layer_idx}.mlp.{proj}.mpo.cores.{c_idx}"
                    if weight_name in ckpt['model_state_dict']:
                        core_shapes.append(ckpt['model_state_dict'][weight_name].shape)
                    else:
                        core_shapes = compute_mpo_core_shapes(out_fac, in_fac, chi_ffn, NUM_CORES)
                        break
                dummy_cores = [torch.zeros(s, device=lin.weight.device, dtype=lin.weight.dtype) for s in core_shapes]
                mpo = MPOLinear(in_f, out_f, dummy_cores, s_vector=s_vec)
                setattr(student_layer.mlp, proj, ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight, skip_svd=True))
            else:
                cores = factor_linear_mpo_custom(
                    lin.weight.cpu().float(), chi_ffn, NUM_CORES, out_fac, in_fac, 
                    s_vec.cpu().float() if s_vec is not None else None, adaptive=True, energy_threshold=0.99
                )
                cleaned_cores = [c.to(device=lin.weight.device, dtype=lin.weight.dtype) for c in cores]
                mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vec)
                setattr(student_layer.mlp, proj, ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight, skip_svd=False))

    print_param_compression_report(teacher_model, student_model)

    if is_resume:
        print(f"\n📂 正在将中断前的数据安全灌入骨架中...")
        student_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        has_nan = False
        for name, param in student_model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                has_nan = True
                break
        if has_nan:
            print("\n" + "!"*70)
            print("🚨 致命警告：你读取的旧存档中包含了 NaN 毒素！")
            print("🚨 请在终端运行命令: rm progressive_layer_checkpoint.pt")
            print("🚨 然后重新启动本脚本！程序已自动拦截启动并退出。")
            print("!"*70 + "\n")
            sys.exit(1)
            
        print(f"✅ 安检通过，未发现 NaN 毒素！将直接从 Layer {start_layer} 继续起飞！")
        del ckpt; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------------
    # 阶段二：累进式局部微调 
    # ---------------------------------------------------------
    if start_layer < num_layers:
        print(f"\n[阶段 2/4] 🧬 启动局部微调 (Teacher Forcing 绝对纯净模式)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 50).select(range(200))
        calib_inputs = [tokenizer(t, return_tensors="pt", max_length=256, truncation=True).input_ids for t in ds["text"]]

        for layer_idx in range(start_layer, num_layers):
            ratio = custom_ratio_map.get(layer_idx, get_u_shape_ratio(layer_idx, num_layers, args.target_ratio))
            # 🚨 保护层直接跳过微调！
            if ratio >= 0.99:
                print(f"\n ⏭️ Layer {layer_idx} 属于首尾保护层，无需局部微调，直接放行！")
                continue

            student_layer = student_model.model.layers[layer_idx]
            teacher_layer = teacher_model.model.layers[layer_idx]
            layer_device = next(student_layer.parameters()).device

            orig_layer_params = sum(p.numel() for p in teacher_layer.mlp.parameters())
            new_layer_params = sum(p.numel() for p in student_layer.mlp.parameters())
            reduced_params = orig_layer_params - new_layer_params
            
            print(f"\n 🎯 正在对齐 Layer {layer_idx} 的局部特征流...")
            print(f"    ✂️ 【瘦身战报】参数从 {orig_layer_params/1e6:.2f}M 降至 {new_layer_params/1e6:.2f}M (甩掉 {reduced_params/1e6:.2f}M 肥肉！)")

            cached_h_in, cached_t_out = [], []
            
            def t_pre_hook(m, args): cached_h_in.append(args[0].detach().float().cpu())
            def t_post_hook(m, args, out): cached_t_out.append(out.detach().float().cpu())
                
            h1 = teacher_layer.mlp.register_forward_pre_hook(t_pre_hook)
            h2 = teacher_layer.mlp.register_forward_hook(t_post_hook)
            
            with torch.no_grad(): 
                for input_ids in tqdm(calib_inputs, desc=f"截获纯净数据", leave=False):
                    teacher_model(input_ids.to(teacher_model.device))
            h1.remove(); h2.remove()

            # =========================================================
            # 🚨 修复后的梯度列表分配方案：彻底解决 dict AttributeError
            # =========================================================
            trainable_params_for_opt = []
            raw_params_for_clip = []
            
            for param in student_model.parameters(): param.requires_grad = False
            
            for proj in ["gate_proj", "up_proj"]:
                wrapper = getattr(student_layer.mlp, proj)
                wrapper.lora_A.requires_grad = True; wrapper.lora_B.requires_grad = True
                for core in wrapper.mpo.cores: core.requires_grad = True
                
                # 给优化器：带 lr 的字典
                trainable_params_for_opt.extend([
                    {"params": wrapper.lora_A, "lr": 1e-4}, 
                    {"params": wrapper.lora_B, "lr": 1e-4}, 
                    {"params": wrapper.mpo.cores, "lr": 1e-5}
                ])
                # 给梯度裁剪：纯张量对象
                raw_params_for_clip.extend([wrapper.lora_A, wrapper.lora_B])
                raw_params_for_clip.extend(wrapper.mpo.cores)
            
            optimizer = torch.optim.AdamW(trainable_params_for_opt, weight_decay=0.01)
            student_layer.train()
            
            for step in range(args.local_steps):
                idx = torch.randint(0, len(cached_h_in), (1,)).item()
                # 🚨 全部使用 Float32 高精度计算！不再用 amp.autocast 降低精度！
                h_in = cached_h_in[idx].to(device=layer_device, dtype=torch.float32)
                t_out = cached_t_out[idx].to(device=layer_device, dtype=torch.float32)
                
                loss = F.mse_loss(student_layer.mlp(h_in), t_out)
                loss.backward()
                
                # 在纯张量列表里检查 NaN
                has_nan_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in raw_params_for_clip)
                if has_nan_grad:
                    optimizer.zero_grad()
                    continue
                    
                # 使用纯张量列表进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(raw_params_for_clip, 1.0)
                optimizer.step(); optimizer.zero_grad()

            del cached_h_in, cached_t_out; torch.cuda.empty_cache(); gc.collect()
            
            print(f"    💾 正在备份进度...")
            tmp_ckpt = PROGRESS_CKPT + ".tmp"
            try:
                torch.save({'next_layer': layer_idx + 1, 'model_state_dict': student_model.state_dict()}, tmp_ckpt)
                os.replace(tmp_ckpt, PROGRESS_CKPT)
                print(f"    ✅ Layer {layer_idx} 处理完毕并已安全固化！")
            except Exception as e:
                print(f"    ⚠️ 保存异常: {e}")
                if os.path.exists(tmp_ckpt): os.remove(tmp_ckpt)

    print("\n[阶段 3/4] 测量累进局部微调后的 PPL...")
    ppl_mid = eval_ppl(student_model, tokenizer)
    print(f"🌟 【阶段二】局部收敛 PPL: {ppl_mid}")

    # ==========================================
    # 🌌 阶段四：全局端到端混合蒸馏
    # ==========================================
    print("\n[阶段 4/4] 🌌 启动端到端联合蒸馏 (E2E Healing)...")
    os.environ["MPO_EVAL_PATH"] = "mpo"; os.environ["MPO_TRAIN_PATH"] = "mpo"
    torch.cuda.empty_cache(); gc.collect()

    student_model = train_healing(
        student_model=student_model, tokenizer=tokenizer, teacher_model=teacher_model, 
        dataset_name="mixed", epochs=1, batch_size=4, accum_steps=2, lr=2e-5, seq_len=1024,  
        save_every_n_steps=500, checkpoint_dir="./healing_checkpoints_mixed_e2e", 
        max_update_steps=args.e2e_steps, resume_from_checkpoint=None 
    )

    ppl_final = eval_ppl(student_model, tokenizer)
    
    print("\n" + "="*70)
    print(" 🏆 究极两阶段微调终极战报")
    print("="*70)
    print(f" 1. 满血原版 PPL          : {ppl_orig:>8.2f}")
    print(f" 2. MPO 累进局部缝合后 PPL : {ppl_mid:>8.2f}")
    print(f" 3. MPO 全局端到端复活 PPL : {ppl_final:>8.2f}")
    print("="*70)

    torch.save(student_model.state_dict(), args.save_path)
    print(f"💾 终极模型已保存至: {args.save_path}")
    if os.path.exists(PROGRESS_CKPT): os.remove(PROGRESS_CKPT)

if __name__ == "__main__":
    main()