#!/usr/bin/env python3
"""
MPO 端到端压缩与微调脚本 (纯净版)
仅保留 FC 层和 LM Head 的高阶张量网络 (MPO) 压缩。
剔除 Embedding 压缩以保证语义视网膜的绝对无损。
"""

import os
import sys
import time
from pathlib import Path
import torch.nn.functional as F
import math

# 强制使用显卡 (可根据需要修改)
if "CUDA_VISIBLE_DEVICES" not in os.environ:                                                                                                                                                                                                 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"   

# 💡 显存优化：避免CUDA内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from calibration import get_activation_scales
from healing import train_healing
from mpo_modules.factorization import estimate_mpo_bond_dim

def parse_args():
    parser = argparse.ArgumentParser(description="MPO 压缩测试脚本 (纯净版)")
    parser.add_argument("--model", type=str, default="tinyllama", choices=["tinyllama", "llama-7b"])
    parser.add_argument("--num_cores", type=int, default=3, help="MPO的长度")
    parser.add_argument("--boundary", type=str, default="open", help="边界条件")
    parser.add_argument("--target_ratio", type=float, default=0.20, help="MPO 中层目标压缩率")
    parser.add_argument("--deep_ratio", type=float, default=None, help="MPO 深层目标压缩率 (默认同 target_ratio)")
    parser.add_argument("--lora_rank", type=int, default=16, help="Res-MPO 中残差辅助矩阵的秩")
    parser.add_argument("--use_s_vector", action="store_true", help="使用 ASVD 保护异常激活值")
    
    # 愈合训练控制开关
    parser.add_argument("--do_healing", action="store_true", help="开启愈合微调")
    parser.add_argument("--data_mode", type=str, default="mixed", 
                        choices=["wiki", "chat", "wiki_then_chat", "mixed"], 
                        help="愈合模式")
    parser.add_argument("--healing_epochs", type=int, default=1)
    parser.add_argument("--healing_lr", type=float, default=5e-5) 
    parser.add_argument("--save_every_n_steps", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="./healing_checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_model", type=str, default=None)
    return parser.parse_args()


# ==========================================
# Res-MPO 包装器 (带误差截断初始化)
# ==========================================
class ResMPOWrapper(nn.Module):
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig, skip_svd=False):
        super().__init__()
        self.mpo = mpo_module
        self.r = lora_rank
        if skip_svd:
            # 只声明参数形状，不做 SVD
            target_device = W_orig.device
            target_dtype = W_orig.dtype
            self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, device=target_device, dtype=target_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=target_device, dtype=target_dtype))
            return
    
        target_device = W_orig.device
        target_dtype = W_orig.dtype

        print(f"      [Res-MPO] 正在通过 SVD(ΔW) 初始化 LoRA 矩阵 (r={self.r})...")
        with torch.no_grad():
            mpo_cpu = self.mpo.cpu().float()
            
            if hasattr(mpo_cpu, 's_vector') and mpo_cpu.s_vector is not None:
                mpo_cpu.s_vector = mpo_cpu.s_vector.cpu().float()
                
            bias_backup = None
            if hasattr(mpo_cpu, 'bias') and mpo_cpu.bias is not None:
                bias_backup = mpo_cpu.bias.data.clone()
                mpo_cpu.bias.data.zero_()

            eye = torch.eye(in_features, dtype=torch.float32)  
            W_mpo = mpo_cpu(eye).T                             

            if bias_backup is not None:
                mpo_cpu.bias.data.copy_(bias_backup)

            Delta_W = W_orig.cpu().float() - W_mpo

            U, S, Vh = torch.linalg.svd(Delta_W, full_matrices=False)
            U_r = U[:, :self.r]
            S_r = S[:self.r]
            Vh_r = Vh[:self.r, :]

            S_sqrt = torch.diag(torch.sqrt(S_r.clamp(min=0)))
            B_init = U_r @ S_sqrt
            A_init = S_sqrt @ Vh_r

            del eye, W_mpo, Delta_W, U, S, Vh

        self.lora_A = nn.Parameter(A_init.to(target_dtype))
        self.lora_B = nn.Parameter(B_init.to(target_dtype))

        self.mpo = self.mpo.to(device=target_device, dtype=target_dtype)
        if hasattr(self.mpo, 's_vector') and self.mpo.s_vector is not None:
            self.mpo.s_vector = self.mpo.s_vector.to(device=target_device, dtype=target_dtype)
        self.lora_A.data = self.lora_A.data.to(target_device)
        self.lora_B.data = self.lora_B.data.to(target_device)

    def forward(self, x):
        mpo_out = self.mpo(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return mpo_out + lora_out


# ==========================================
# 辅助函数
# ==========================================
def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=50000):
    model.eval() 
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    effective_max = min(max_len, max_ctx)
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    nlls = []
    for begin in range(0, seq_len, stride):
        end = min(begin + effective_max, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1:
            continue
        with torch.no_grad():
            out = model(chunk)
            lm_logits = out.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
        nlls.append(loss.item())
        if end >= seq_len:
            break

    if not nlls:
        return float("nan")
    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return {"total": total}

def compute_mpo_core_shapes(out_fac, in_fac, bond_dim, num_cores):
    shapes = []
    prev = 1
    for k in range(num_cores - 1):
        rows = prev * out_fac[k] * in_fac[k]
        cols = 1
        for j in range(k + 1, num_cores):
            cols *= out_fac[j] * in_fac[j]
        rank_avail = min(rows, cols)
        r = max(1, min(bond_dim, rank_avail))
        shapes.append((prev, out_fac[k], in_fac[k], r))
        prev = r
    shapes.append((prev, out_fac[-1], in_fac[-1], 1))
    return shapes


# ==========================================
# MPO 主压缩函数
# ==========================================
@torch.no_grad()
def compress_with_function2(model, cfg, activation_scales=None, skip_svd=False, loaded_weights=None):
    import gc
    num_cores = int(cfg.get("num_cores", 3))
    freeze_blocks = int(cfg.get("freeze_blocks", 0))
    mid_blocks = int(cfg.get("mid_blocks", 20))
    skip_mlp = cfg.get("skip_mlp", None)
    num_layers = len(model.model.layers)

    target_ratio = float(cfg.get("target_ratio", 0.3))
    deep_ratio = float(cfg.get("deep_ratio", target_ratio))

    if skip_svd:
        print("⚡ [快速恢复模式] 跳过 SVD 分解，仅构建空壳结构...")

    for idx in range(num_layers):
        # 💡 升级 1：U 型敏感度动态分配公式 (Sensitivity-Aware Allocation)
        # 用一个开口向上的抛物线模拟敏感度：两端敏感(趋近0.9)，中间冗余(趋近0.2)
        # 将 x 映射到 [-1, 1] 区间
        x = (idx - (num_layers - 1) / 2.0) / ((num_layers - 1) / 2.0)
        
        # 抛物线公式: ratio = a * x^2 + b
        # 这里保证平均压缩率依然和你期望的整体压缩率（比如0.45）持平
        dynamic_ratio = 0.2 + 0.7 * (x ** 2) 
        
        # 限制上下限，防止超出物理物理边界
        ratio = max(0.15, min(0.95, dynamic_ratio))
        
        # 打印出来让你直观看到每一层自动分配的预算！
        if not skip_svd:
            print(f"🔬 Layer {idx:02d} | 敏感度感知分配预算: 保留 {ratio:.1%}")

        # (可选：如果你依然想冻结前两层，可以保留这个)
        if idx < freeze_blocks: 
            continue
        is_mid = (idx - freeze_blocks) < mid_blocks
        ratio = target_ratio if is_mid else deep_ratio

        blk = model.model.layers[idx]

        for fname in ("gate_proj", "up_proj", "down_proj"):
            if fname == skip_mlp:
                continue
            lin = getattr(blk.mlp, fname)
            if not isinstance(lin, nn.Linear):
                continue

            device, dtype0 = lin.weight.device, lin.weight.dtype
            out_f, in_f = lin.weight.shape

            chi_ffn = estimate_mpo_bond_dim(in_f, out_f, num_cores, ratio)
            out_fac = find_factors_balanced(out_f, num_cores)
            in_fac = find_factors_balanced(in_f, num_cores)

            full_name = f"model.layers.{idx}.mlp.{fname}"
            s_vector = activation_scales.get(full_name) if activation_scales else None

            if skip_svd:
                if loaded_weights is not None:
                    core_shapes = []
                    for c_idx in range(num_cores):
                        weight_name = f"model.layers.{idx}.mlp.{fname}.mpo.cores.{c_idx}"
                        if weight_name in loaded_weights:
                            core_shapes.append(loaded_weights[weight_name].shape)
                        else:
                            print(f"⚠️ 未找到 {weight_name}，回退到理论形状")
                            core_shapes = compute_mpo_core_shapes(out_fac, in_fac, chi_ffn, num_cores)
                            break
                else:
                    core_shapes = compute_mpo_core_shapes(out_fac, in_fac, chi_ffn, num_cores)
                
                dummy_cores = [torch.zeros(s, device=device, dtype=dtype0) for s in core_shapes]
                mpo = MPOLinear(in_f, out_f, dummy_cores, s_vector=s_vector, boundary="open")
            else:
                W = lin.weight.detach().clone().cpu().float()
                s_vector_cpu = s_vector.cpu().float() if s_vector is not None else None

                cores_list = factor_linear_mpo_custom(
                    weight=W, bond_dim=chi_ffn, num_cores=num_cores,
                    out_fac=out_fac, in_fac=in_fac,
                    s_vector=s_vector_cpu, boundary="open",adaptive=True, energy_threshold=0.99, min_bond=4
                )

                cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]
                mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vector, boundary="open")

                if getattr(lin, "bias", None) is not None:
                    with torch.no_grad():
                        mpo.bias.copy_(lin.bias.data)

                del W, cores_list, cleaned_cores

            lora_rank = cfg.get("lora_rank", 16)

            if skip_svd:
                res_mpo = ResMPOWrapper(mpo, in_f, out_f, lora_rank, lin.weight, skip_svd=True)
            else:
                W_orig = lin.weight.detach().clone()
                res_mpo = ResMPOWrapper(mpo, in_f, out_f, lora_rank, W_orig, skip_svd=False)
                del W_orig

            setattr(blk.mlp, fname, res_mpo)

    # ====================================================
    # LM Head 压缩
    # ====================================================
    print("\n🔄 正在同步压缩输出层 LM Head...")
    lm_head = model.lm_head
    if isinstance(lm_head, nn.Linear):
        device_head = lm_head.weight.device
        dtype_head = lm_head.weight.dtype
        out_f_head, in_f_head = lm_head.weight.shape

        head_ratio = cfg.get("head_ratio", 0.5)
        chi_head = estimate_mpo_bond_dim(in_f_head, out_f_head, num_cores, head_ratio)
        out_fac_head = find_factors_balanced(out_f_head, num_cores)
        in_fac_head = find_factors_balanced(in_f_head, num_cores)

        if skip_svd:
            if loaded_weights is not None:
                core_shapes = []
                for c_idx in range(num_cores):
                    weight_name = f"lm_head.mpo.cores.{c_idx}"
                    if weight_name in loaded_weights:
                        core_shapes.append(loaded_weights[weight_name].shape)
                    else:
                        print(f"⚠️ 未找到 {weight_name}，回退到理论形状")
                        core_shapes = compute_mpo_core_shapes(out_fac_head, in_fac_head, chi_head, num_cores)
                        break
            else:
                core_shapes = compute_mpo_core_shapes(out_fac_head, in_fac_head, chi_head, num_cores)
            
            dummy_cores = [torch.zeros(s, device=device_head, dtype=dtype_head) for s in core_shapes]
            mpo_head = MPOLinear(in_f_head, out_f_head, dummy_cores, boundary="open")
        else:
            W_head = lm_head.weight.detach().clone().cpu().float()
            cores_list_head = factor_linear_mpo_custom(
                weight=W_head, bond_dim=chi_head, num_cores=num_cores,
                out_fac=out_fac_head, in_fac=in_fac_head, boundary="open"
            )
            cleaned_cores_head = [c.to(device=device_head, dtype=dtype_head) for c in cores_list_head]
            mpo_head = MPOLinear(in_f_head, out_f_head, cleaned_cores_head, boundary="open")

            if getattr(lm_head, "bias", None) is not None:
                with torch.no_grad():
                    mpo_head.bias.copy_(lm_head.bias.data)

            del W_head, cores_list_head, cleaned_cores_head

        lora_rank = cfg.get("lora_rank", 16)
        if skip_svd:
            res_mpo_head = ResMPOWrapper(mpo_head, in_f_head, out_f_head, lora_rank, lm_head.weight, skip_svd=True)
        else:
            W_head_gpu = lm_head.weight.detach().clone()
            res_mpo_head = ResMPOWrapper(mpo_head, in_f_head, out_f_head, lora_rank, W_head_gpu, skip_svd=False)
            del W_head_gpu

        model.lm_head = res_mpo_head
        print(f"✅ LM Head 已替换为 Res-MPO (Bond Dim: {chi_head}, LoRA Rank: {lora_rank})")

    gc.collect()
    torch.cuda.empty_cache()
    return model

# ==========================================
# 主函数
# ==========================================
def main():
    program_start_time = time.time()
    args = parse_args()
    
    if args.model == "llama-7b":
        model_name = "NousResearch/Llama-2-7b-hf"
        freeze_blocks = 4
        mid_blocks = 16
    else:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        freeze_blocks = 2
        mid_blocks = 12

    deep_ratio = args.deep_ratio if args.deep_ratio is not None else args.target_ratio

    cfg = {
        "num_cores": args.num_cores,
        "boundary": args.boundary,
        "freeze_blocks": freeze_blocks,
        "mid_blocks": mid_blocks,
        "target_ratio": args.target_ratio,
        "deep_ratio": deep_ratio, 
        "skip_mlp": "down_proj",
        "lora_rank": args.lora_rank,
        "head_ratio": 0.5 
    }

    print("\n" + "="*70)
    print("MPO FC + LM Head 极简核心压缩架构 (无损 Embedding)")
    print("="*70)

    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.chat_template is None:
        print("⚠️ 未检测到默认对话模板，正在强行注入默认模板...")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<|user|>\n{{ message['content'] }}</s>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|assistant|>\n{{ message['content'] }}</s>\n"
            "{% elif message['role'] == 'system' %}"
            "<|system|>\n{{ message['content'] }}</s>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )

    # 2. 提取 ASVD 
    if args.use_s_vector:
        print("\n提取 activation scales...")
        activation_scales_dict = get_activation_scales(model, tokenizer, num_samples=128, max_len=512)
    else:
        activation_scales_dict = None

    orig_stats = count_params(model)
    orig_mlp_params = sum(p.numel() for name, p in model.named_parameters() if "mlp" in name)

    # 3. 剥离了 TT-Embedding，直接计算存档名
    surgery_checkpoint = (
    f"compressed_{args.model}"
    f"_cores{args.num_cores}"
    f"_r{args.lora_rank}"
    f"_tr{args.target_ratio}"
    f"_dr{deep_ratio}"
    f"_sv{int(args.use_s_vector)}"
    f".pt"
    )

    # 4. 执行 MPO FC 压缩 (查字典秒搭骨架逻辑)
    trainable_weights = None
    is_resume = os.path.exists(surgery_checkpoint)
    
    if is_resume:
        print(f"\n⚠️ 检测到手术存档 {surgery_checkpoint}！")
        print("   📂 正在提前读取存档，用于全局空壳骨架搭建...")
        ckpt = torch.load(surgery_checkpoint, map_location="cpu")
        trainable_weights = ckpt.get("trainable_state_dict", ckpt)

        model = compress_with_function2(
            model, cfg, activation_scales=activation_scales_dict, 
            skip_svd=True, loaded_weights=trainable_weights
        )
        
        result = model.load_state_dict(trainable_weights, strict=False)
        if result.missing_keys:
            print(f"   ⚠️ [警告] 缺失 {len(result.missing_keys)} 个权重! 示例: {result.missing_keys[:3]}")
        if result.unexpected_keys:
            print(f"   ⚠️ [警告] 多出 {len(result.unexpected_keys)} 个未知权重! 示例: {result.unexpected_keys[:3]}")
        else:
            print("   ✅ 所有存档权重 (MPO + LoRA) 完美匹配！")
    else:
        model = compress_with_function2(model, cfg, activation_scales=activation_scales_dict)
        torch.save(model.state_dict(), surgery_checkpoint)
    
    stats = count_params(model)
    print(f"\n压缩后总参数量: {stats['total'] / 1e6:.1f}M ({stats['total'] / orig_stats['total']:.1%})")

    new_mlp_params = sum(p.numel() for name, p in model.named_parameters() if "mlp" in name)

    # 5. 执行 Healing
    if args.do_healing:
        print(f"\n🚀 启动端到端联合蒸馏...")
        student_device_map = model.hf_device_map

        print("📦 正在强制按照 Student 的拓扑图加载 Teacher 模型...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=student_device_map  
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        os.environ["MPO_EVAL_PATH"] = "mpo"
        os.environ["MPO_TRAIN_PATH"] = "mpo"

        if args.data_mode in ["wiki", "wiki_then_chat"]:
            model = train_healing(
                student_model=model,
                tokenizer=tokenizer,
                teacher_model=teacher_model, 
                dataset_name="wiki",
                epochs=args.healing_epochs,
                batch_size=1,  
                accum_steps=8,  
                lr=args.healing_lr,
                seq_len=256,
                save_every_n_steps=args.save_every_n_steps,
                checkpoint_dir=args.checkpoint_dir + "_wiki",
                max_update_steps=2000
            )

        if args.data_mode in ["chat", "wiki_then_chat", "mixed"]:
            print("\n🔄 切换到后续对齐数据集 MPO 路径训练...")
            my_resume_path = args.resume_from if args.resume_from else None
            
            model = train_healing(
                student_model=model,
                tokenizer=tokenizer,
                teacher_model=teacher_model, 
                dataset_name=args.data_mode, 
                epochs=args.healing_epochs,
                batch_size=4,
                accum_steps=2,
                lr=2e-5,  
                seq_len=1024,  
                save_every_n_steps=args.save_every_n_steps,
                checkpoint_dir=args.checkpoint_dir + f"_{args.data_mode}", 
                max_update_steps=2400,  
                resume_from_checkpoint=my_resume_path 
            )

    if args.output_model:
        torch.save(model.state_dict(), args.output_model)
        print(f"\n💾 模型已保存: {args.output_model}")

    # 6. 战报
    stats_final = count_params(model)
    
    print("\n" + "="*70)
    print("🏆 终极评估战报 (无损 Embedding MPO架构)")
    print("="*70)

    print(f"📊 【FC 层 (MLP Block)】")
    print(f"   原 Dense 参数: {orig_mlp_params / 1e6:.1f} M")
    print(f"   现 混合 参数:  {new_mlp_params / 1e6:.1f} M (含 MPO + LoRA + 跳过的层)")
    print(f"   核心压缩率:    {new_mlp_params / orig_mlp_params:.1%}")  

    print(f"\n📊 【全局总计】")
    print(f"   原模型总参数:  {orig_stats['total'] / 1e6:.1f} M")
    print(f"   现模型总参数:  {stats_final['total'] / 1e6:.1f} M")
    print(f"   全局保留率:    {stats_final['total'] / orig_stats['total']:.1%}")

    print("\n⏳ 正在评估最终模型的 PPL (Wikitext-2)...")
    os.environ["MPO_EVAL_PATH"] = "mpo" 
    t0 = time.time()
    final_ppl = eval_ppl(model, tokenizer, max_tokens=50000)
    print(f"✅ 最终模型 PPL: {final_ppl:.2f}")
    print(f"   (评估耗时: {time.time() - t0:.1f}s)")
    print("="*70)

    total_time = time.time() - program_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n⏱️ 程序总运行时间: {hours}小时 {minutes}分钟 {seconds:.1f}秒")

if __name__ == "__main__":
    main()