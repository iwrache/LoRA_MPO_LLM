#!/usr/bin/env python3
"""
完整对比MPO分解的不同配置：
1. 原始模型
2. 函数1: factor_linear_mpo (封装层) - 有截断, 无 s_vector
3. 函数2: factor_linear_mpo_custom (核心算法) - 有截断, 有 s_vector (activation-aware)
4. 函数3: 满秩MPO (无截断) - 测试SVD不截断的效果
"""

import json
import os
import sys
import time
from pathlib import Path

# 强制使用 4 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from mpo_modules.core import MPOLinear
from mpo_modules.model_utils import replace_llama_linears_by_maps
from mpo_modules.factorization import factor_linear_mpo as factor_linear_mpo_original
from mpo_modules.helpers import find_factors_balanced as find_factors_balanced_orig
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from calibration import get_activation_scales
from healing import train_healing

def parse_args():
    """解析命令行输入的参数"""
    parser = argparse.ArgumentParser(description="MPO 分解参数化测试脚本")
    parser.add_argument("--model", type=str, default="tinyllama", choices=["tinyllama", "llama-7b"], help="选择模型: tinyllama (1.1B) 或 llama-7b (7B)")
    parser.add_argument("--num_cores", type=int, default=3, help="MPO的长度 (如2, 3, 4)")
    parser.add_argument("--boundary", type=str, default="open", choices=["open", "periodic"], help="边界条件")
    parser.add_argument("--target_ratio", type=float, default=0.20, help="目标压缩率")
    # ASVD 白化保护开关
    parser.add_argument("--use_s_vector", action="store_true", help="加上此参数则提取并使用 s_vector 保护异常激活值")
    # ==========================================
    # [新增]：愈合训练控制开关
    # ==========================================
    parser.add_argument("--do_healing", action="store_true", help="加上这个参数则开启愈合微调，不加则跳过")
    parser.add_argument("--healing_epochs", type=int, default=1, help="愈合训练的 Epoch 数")
    parser.add_argument("--healing_lr", type=float, default=5e-5, help="愈合训练的学习率")
    # Checkpoint 相关参数
    parser.add_argument("--save_every_n_steps", type=int, default=200, help="每 N 步保存 checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="./healing_checkpoints", help="Checkpoint 保存目录")
    parser.add_argument("--resume_from", type=str, default=None, help="从 checkpoint 恢复训练 (路径)")
    parser.add_argument("--output_model", type=str, default=None, help="最终模型保存路径")
    # Bond dimension 扩展参数
    parser.add_argument("--expand_bond_factor", type=float, default=None, help="扩展 bond dimension 的倍数 (例如 1.5 表示扩展50%%)，None 表示不扩展")
    parser.add_argument("--expand_noise_std", type=float, default=0.01, help="扩展部分填充的正态分布噪音标准差")
    return parser.parse_args()

def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=50000):
    """Evaluate perplexity on Wikitext-2 test."""
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


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """生成文本"""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def count_params(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    mpo_total = 0
    dense_total = 0
    mpo_layers = 0
    for name, mod in model.named_modules():
        if isinstance(mod, MPOLinear):
            mpo_params = sum(p.numel() for p in mod.parameters())
            dense_params = mod.out_f * mod.in_f
            mpo_total += mpo_params
            dense_total += dense_params
            mpo_layers += 1
    return {
        "total": total,
        "mpo_layers": mpo_layers,
        "mpo_params": mpo_total,
        "dense_equivalent": dense_total,
    }


@torch.no_grad()
def compress_with_function1(model, cfg):
    """使用封装层函数 - 有截断 (不使用 s_vector / activation-aware)"""
    num_cores = int(cfg.get("num_cores", 3))
    freeze_blocks = int(cfg.get("freeze_blocks", 0))
    mid_blocks = int(cfg.get("mid_blocks", 20))
    skip_mlp = cfg.get("skip_mlp", None)
    num_layers = len(model.model.layers)

    chi_attn, core_attn, chi_ffn, core_ffn = {}, {}, {}, {}
    use_ratio = str(cfg.get("mode", "fixed")).strip().lower() == "ratio"
    if use_ratio:
        from mpo_modules.factorization import estimate_mpo_bond_dim
        target_ratio = float(cfg.get("target_ratio", 0.3))
        deep_ratio = float(cfg.get("deep_ratio", target_ratio))
    else:
        mid_chi = int(cfg.get("mid_chi", 100))
        deep_chi = int(cfg.get("deep_chi", 40))

    for idx in range(num_layers):
        if idx < freeze_blocks:
            continue
        is_mid = (idx - freeze_blocks) < mid_blocks
        core_attn[idx] = num_cores
        core_ffn[idx] = num_cores

        if use_ratio:
            ratio = target_ratio if is_mid else deep_ratio
            for fname in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(model.model.layers[idx].self_attn, fname)
                if isinstance(lin, nn.Linear):
                    chi_attn[idx] = estimate_mpo_bond_dim(lin.in_features, lin.out_features, num_cores, ratio)
                    break
            for fname in ("gate_proj", "up_proj", "down_proj"):
                if fname == skip_mlp:
                    continue
                lin = getattr(model.model.layers[idx].mlp, fname)
                if isinstance(lin, nn.Linear):
                    chi_ffn[idx] = estimate_mpo_bond_dim(lin.in_features, lin.out_features, num_cores, ratio)
                    break
        else:
            chi = mid_chi if is_mid else deep_chi
            chi_attn[idx] = chi
            chi_ffn[idx] = chi

    model, _ = replace_llama_linears_by_maps(model, chi_attn, core_attn, chi_ffn, core_ffn, skip_mlp=skip_mlp)
    return model


@torch.no_grad()
def compress_with_function2(model, cfg, use_func1_factors=False, activation_scales=None):
    """
    使用核心算法函数 - 有截断 (支持 activation_scales / s_vector)

    Args:
        use_func1_factors: 如果为True，使用mpo_modules.helpers.find_factors_balanced
                          来匹配函数1的因子分解（从而获得相似的压缩率）
        activation_scales: 校准得到的激活缩放字典，如果提供则使用 s_vector
    """
    import gc
    num_cores = int(cfg.get("num_cores", 3))
    freeze_blocks = int(cfg.get("freeze_blocks", 0))
    mid_blocks = int(cfg.get("mid_blocks", 20))
    skip_mlp = cfg.get("skip_mlp", None)
    num_layers = len(model.model.layers)

    use_ratio = str(cfg.get("mode", "fixed")).strip().lower() == "ratio"
    if use_ratio:
        target_ratio = float(cfg.get("target_ratio", 0.3))
        deep_ratio = float(cfg.get("deep_ratio", target_ratio))

    # 选择使用哪个find_factors_balanced
    if use_func1_factors:
        factor_fn = find_factors_balanced_orig  # 使用函数1的因子分解
    else:
        factor_fn = find_factors_balanced  # 使用函数2原来的

    for idx in range(num_layers):
        if idx < freeze_blocks:
            continue

        is_mid = (idx - freeze_blocks) < mid_blocks

        if use_ratio:
            ratio = target_ratio if is_mid else deep_ratio
            from mpo_modules.factorization import estimate_mpo_bond_dim
            # 为 Attention 层计算 chi (使用 q_proj 的尺寸 2048x2048)
            chi_attn = estimate_mpo_bond_dim(2048, 2048, num_cores, ratio)
            # 为 FFN 层计算 chi (使用 gate_proj 的实际尺寸 5632x2048)
            chi_ffn = estimate_mpo_bond_dim(5632, 2048, num_cores, ratio)
        else:
            chi_attn = int(cfg.get("mid_chi", 100)) if is_mid else int(cfg.get("deep_chi", 40))
            chi_ffn = chi_attn

        blk = model.model.layers[idx]

        # Self-Attn - 使用 chi_attn
        for fname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            # 修改为：跳过 Q 和 K
            if fname in ("q_proj", "k_proj"):
                continue
            lin = getattr(blk.self_attn, fname)
            if not isinstance(lin, nn.Linear):
                continue

            W = lin.weight.detach().clone()
            out_f, in_f = W.shape
            device = lin.weight.device
            dtype0 = lin.weight.dtype

            out_fac = factor_fn(out_f, num_cores)
            in_fac = factor_fn(in_f, num_cores)

            # 获取当前层的 s_vector (如果提供了 activation_scales)
            full_name = f"model.layers.{idx}.self_attn.{fname}"
            s_vector = activation_scales.get(full_name) if activation_scales else None
            
            boundary_condition = cfg.get("boundary", "open")
            # [修改]：把 boundary 传给底层的 factor 函数
            cores_list = factor_linear_mpo_custom(
                weight=W, bond_dim=chi_attn, num_cores=num_cores,
                out_fac=out_fac, in_fac=in_fac, 
                s_vector=s_vector, 
                boundary=boundary_condition,  # <--- 传下去！
                noise_scale=1e-5
            )
            cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]

            mpo = MPOLinear(
                in_f, out_f, cleaned_cores, 
                boundary=boundary_condition,  # <--- 传下去！
                s_vector=s_vector
            )

            if getattr(lin, "bias", None) is not None and getattr(mpo, "bias", None) is not None:
                with torch.no_grad():
                    mpo.bias.copy_(lin.bias.data)
            setattr(blk.self_attn, fname, mpo)

        # FFN - 使用 chi_ffn (更高的 bond_dim，匹配函数1)
        for fname in ("gate_proj", "up_proj", "down_proj"):
            if fname == skip_mlp:
                continue
            lin = getattr(blk.mlp, fname)
            if not isinstance(lin, nn.Linear):
                continue

            W = lin.weight.detach().clone()
            out_f, in_f = W.shape
            device = lin.weight.device
            dtype0 = lin.weight.dtype

            out_fac = factor_fn(out_f, num_cores)
            in_fac = factor_fn(in_f, num_cores)

            # 获取当前层的 s_vector (如果提供了 activation_scales)
            full_name = f"model.layers.{idx}.mlp.{fname}"
            s_vector = activation_scales.get(full_name) if activation_scales else None

            cores_list = factor_linear_mpo_custom(weight=W, bond_dim=chi_ffn, num_cores=num_cores,
                                                  out_fac=out_fac, in_fac=in_fac, s_vector=s_vector)
            cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]
            mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vector, boundary="open")

            if getattr(lin, "bias", None) is not None and getattr(mpo, "bias", None) is not None:
                with torch.no_grad():
                    mpo.bias.copy_(lin.bias.data)
            setattr(blk.mlp, fname, mpo)

    gc.collect()
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def compress_full_rank(model, cfg):
    """
    使用满秩MPO分解（无截断）
    通过设置 bond_dim = 999999 来实现不截断
    """
    import gc
    num_cores = int(cfg.get("num_cores", 3))
    freeze_blocks = int(cfg.get("freeze_blocks", 0))
    mid_blocks = int(cfg.get("mid_blocks", 20))
    skip_mlp = cfg.get("skip_mlp", None)
    num_layers = len(model.model.layers)

    FULL_RANK_BOND = 999999  # 满秩，不截断

    for idx in range(num_layers):
        if idx < freeze_blocks:
            continue

        blk = model.model.layers[idx]

        # Self-Attn - 满秩
        for fname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(blk.self_attn, fname)
            if not isinstance(lin, nn.Linear):
                continue

            W = lin.weight.detach().clone()
            out_f, in_f = W.shape
            device = lin.weight.device
            dtype0 = lin.weight.dtype

            out_fac = find_factors_balanced(out_f, num_cores)
            in_fac = find_factors_balanced(in_f, num_cores)

            # 使用满秩 bond_dim
            cores_list = factor_linear_mpo_custom(weight=W, bond_dim=FULL_RANK_BOND,
                                                  num_cores=num_cores,
                                                  out_fac=out_fac, in_fac=in_fac)
            cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]
            mpo = MPOLinear(in_f, out_f, cleaned_cores, boundary="open")

            if getattr(lin, "bias", None) is not None and getattr(mpo, "bias", None) is not None:
                with torch.no_grad():
                    mpo.bias.copy_(lin.bias.data)
            setattr(blk.self_attn, fname, mpo)

        # FFN - 满秩
        for fname in ("gate_proj", "up_proj", "down_proj"):
            if fname == skip_mlp:
                continue
            lin = getattr(blk.mlp, fname)
            if not isinstance(lin, nn.Linear):
                continue

            W = lin.weight.detach().clone()
            out_f, in_f = W.shape
            device = lin.weight.device
            dtype0 = lin.weight.dtype

            out_fac = find_factors_balanced(out_f, num_cores)
            in_fac = find_factors_balanced(in_f, num_cores)

            # 使用满秩 bond_dim
            cores_list = factor_linear_mpo_custom(weight=W, bond_dim=FULL_RANK_BOND,
                                                  num_cores=num_cores,
                                                  out_fac=out_fac, in_fac=in_fac)
            cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]
            mpo = MPOLinear(in_f, out_f, cleaned_cores, boundary="open")

            if getattr(lin, "bias", None) is not None and getattr(mpo, "bias", None) is not None:
                with torch.no_grad():
                    mpo.bias.copy_(lin.bias.data)
            setattr(blk.mlp, fname, mpo)

    gc.collect()
    torch.cuda.empty_cache()
    return model


def print_section(title, char="="):
    """打印分隔符"""
    print(f"\n{char * 70}")
    print(title)
    print(f"{char * 70}")


def expand_mpo_cores(model, expand_factor, noise_std=0.01):
    """
    扩展 MPO 的 bond dimension。
    将每个 core 的 bond 维度扩展 expand_factor 倍，
    原始张量放在左上角，其余位置用正态分布噪音填充。

    Args:
        model: 包含 MPOLinear 层的模型
        expand_factor: bond dimension 扩展倍数
        noise_std: 填充噪音的标准差
    """
    import torch
    from mpo_modules.core import MPOLinear

    if expand_factor is None or expand_factor <= 1.0:
        return model

    print(f"\n🔧 扩展 MPO bond dimension: {expand_factor}x")
    print(f"   填充噪音标准差: {noise_std}")

    expanded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, MPOLinear):
            new_cores = []
            for i, core in enumerate(module.cores):
                # core 形状: (r_prev, o_k, i_k, r_next)
                r_prev, o_k, i_k, r_next = core.shape

                # 计算新的 bond dimension
                new_r_prev = max(1, int(r_prev * expand_factor))
                new_r_next = max(1, int(r_next * expand_factor))

                # 对于第一个和最后一个 core，保持边界维度为 1（open boundary）或统一扩展（periodic）
                if module.boundary == "open":
                    if i == 0:
                        new_r_prev = 1
                    if i == len(module.cores) - 1:
                        new_r_next = 1

                # 创建新的 core 张量
                new_core = torch.randn(new_r_prev, o_k, i_k, new_r_next) * noise_std

                # 将原始张量放在左上角
                new_core[:r_prev, :, :, :r_next] = core.data

                new_cores.append(torch.nn.Parameter(new_core.contiguous()))

            # 替换 cores
            module.cores = torch.nn.ParameterList(new_cores)
            expanded_count += 1

    print(f"✅ 已扩展 {expanded_count} 个 MPO 层的 bond dimension")
    return model


def main():
    args = parse_args()  # <--- [新增] 获取命令行参数

    # ==========================================
    # 根据参数选择模型和配置
    # ==========================================
    if args.model == "llama-7b":
        model_name = "NousResearch/Llama-2-7b-hf"  # 无需权限的开源版本
        # Llama-7B 有 32 层，需要调整块配置
        freeze_blocks = 4
        mid_blocks = 16
    else:  # tinyllama (默认)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        freeze_blocks = 2
        mid_blocks = 12

    cfg = {
        "mode": "ratio",
        "num_cores": args.num_cores,       # <--- 动态参数
        "boundary": args.boundary,         # <--- 动态参数
        "freeze_blocks": freeze_blocks,    # <--- 根据模型调整
        "mid_blocks": mid_blocks,          # <--- 根据模型调整
        "target_ratio": args.target_ratio, # <--- 动态参数
        "deep_ratio": 0.40,
        "skip_mlp": "down_proj",
    }

    print_section("MPO 分解完整对比测试 (含满秩无截断)", "=")
    print(f"\n模型: {model_name}")
    print(f"模型类型: {args.model}")
    print(f"配置: {cfg}")

    # ==========================================
    # 加载原始模型
    # ==========================================
    print_section("【步骤1】加载原始模型")
    t0 = time.time()
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"加载耗时: {time.time() - t0:.1f}s")

    orig_stats = count_params(model_orig)
    print(f"\n参数量统计:")
    print(f"  总参数量: {orig_stats['total'] / 1e6:.1f}M")

    # ==========================================
    # 测试Prompt
    # ==========================================
    test_prompts = [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "The best way to learn programming is",
    ]

    print_section("【步骤2】原始模型输出")
    print("\n测试Prompts:")
    orig_outputs = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        output = generate_text(model_orig, tokenizer, prompt, max_new_tokens=30)
        orig_outputs.append(output)
        print(f"   Output: {output}")

    # ==========================================
    # 原始模型PPL
    # ==========================================
    print_section("【步骤3】原始模型PPL评估")
    print("(限制50K tokens以加快测试)")
    t0 = time.time()
    orig_ppl = eval_ppl(model_orig, tokenizer, max_tokens=50000)
    print(f"原始模型PPL: {orig_ppl:.2f}")
    print(f"评估耗时: {time.time() - t0:.1f}s")

    del model_orig
    torch.cuda.empty_cache()

    results = {
        "original": {"ppl": orig_ppl, "stats": orig_stats, "outputs": orig_outputs}
    }

    # ==========================================
    # 测试1: 函数1 (有截断)
    # ==========================================
    print_section("【步骤4】测试1: factor_linear_mpo (有截断, 无 s_vector)")

    print("重新加载模型...")
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )

    print("\n开始压缩...")
    t0 = time.time()
    model1 = compress_with_function1(model1, cfg)
    compress_time1 = time.time() - t0

    stats1 = count_params(model1)
    print(f"\n压缩耗时: {compress_time1:.1f}s")
    print(f"参数量统计:")
    print(f"  总参数量: {stats1['total'] / 1e6:.1f}M ({stats1['total'] / orig_stats['total']:.1%})")
    print(f"  MPO层数: {stats1['mpo_layers']}")
    print(f"  MPO压缩率: {stats1['mpo_params'] / stats1['dense_equivalent']:.1%}")

    print("\n生成输出:")
    os.environ["MPO_EVAL_PATH"] = "dense"
    func1_outputs = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        output = generate_text(model1, tokenizer, prompt, max_new_tokens=30)
        func1_outputs.append(output)
        print(f"   Output: {output}")

    print("\nPPL评估...")
    t0 = time.time()
    ppl1 = eval_ppl(model1, tokenizer, max_tokens=50000)
    print(f"函数1 PPL: {ppl1:.2f} (ΔPPL: {ppl1 - orig_ppl:+.2f})")
    print(f"评估耗时: {time.time() - t0:.1f}s")

    results["func1_truncated"] = {"ppl": ppl1, "stats": stats1, "outputs": func1_outputs, "time": compress_time1}

    del model1
    torch.cuda.empty_cache()

    # ==========================================
    # 测试2: 函数2 (调整版，使用函数1因子分解)
    # ==========================================
    print_section("【步骤5】测试2: factor_linear_mpo_custom (调整版, 使用 s_vector)")

    print("重新加载模型...")
    print("注意: 使用 use_func1_factors=True 来匹配函数1的因子分解，从而获得相近参数量")
    print("注意: 函数2使用 activation_scales (s_vector) 进行 activation-aware MPO 分解")
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )

    # 动态决定是否抓取 activation_scales
    if args.use_s_vector:
        print("\n获取 activation scales (开启 ASVD)...")
        activation_scales_dict = get_activation_scales(model2, tokenizer, num_samples=128, max_len=512)
    else:
        print("\n跳过 activation scales 提取 (--use_s_vector 未开启)...")
        activation_scales_dict = None

    print("\n开始压缩 (使用 s_vector)...")
    t0 = time.time()
    model2 = compress_with_function2(model2, cfg, use_func1_factors=True, activation_scales=activation_scales_dict)
    compress_time2 = time.time() - t0

    stats2 = count_params(model2)
    print(f"\n压缩耗时: {compress_time2:.1f}s")
    print(f"参数量统计:")
    print(f"  总参数量: {stats2['total'] / 1e6:.1f}M ({stats2['total'] / orig_stats['total']:.1%})")
    print(f"  MPO层数: {stats2['mpo_layers']}")
    print(f"  MPO压缩率: {stats2['mpo_params'] / stats2['dense_equivalent']:.1%}")

    # ==========================================
    # [新增]：扩展 bond dimension（如果指定）
    # ==========================================
    if args.expand_bond_factor is not None:
        print("\n" + "="*70)
        print(f"扩展 MPO bond dimension: {args.expand_bond_factor}x")
        print("="*70)
        model2 = expand_mpo_cores(model2, args.expand_bond_factor, args.expand_noise_std)
        # 重新统计参数
        stats2 = count_params(model2)
        print(f"\n扩展后参数量统计:")
        print(f"  总参数量: {stats2['total'] / 1e6:.1f}M ({stats2['total'] / orig_stats['total']:.1%})")
        print(f"  MPO层数: {stats2['mpo_layers']}")
        print(f"  MPO压缩率: {stats2['mpo_params'] / stats2['dense_equivalent']:.1%}")

    # ========================================================
    # [修改]：根据命令行参数，决定是否执行愈合训练！
    # ========================================================
    if args.do_healing:
        print(f"\n🚀 启动 MPO 全局愈合微调 (Epochs: {args.healing_epochs}, LR: {args.healing_lr})...")
        os.environ["MPO_EVAL_PATH"] = "mpo"
        os.environ["MPO_TRAIN_PATH"] = "dense"  # 强制走 dense 前向传播避免 autograd 断裂

        # 将动态参数传入 healing 函数
        model2 = train_healing(
            model2,
            tokenizer,
            epochs=args.healing_epochs,
            batch_size=1,
            accum_steps=32,
            lr=args.healing_lr,
            seq_len=256,
            save_every_n_steps=args.save_every_n_steps,
            resume_from_checkpoint=args.resume_from,
            checkpoint_dir=args.checkpoint_dir
        )

    else:
        print("\n⏩ 跳过全局愈合微调 (--do_healing 未开启)，直接使用 ASVD 截断后的权重进行测试。")

    # 保存最终模型（如果指定了输出路径）- 无论是否做 healing 都会保存
    if args.output_model:
        torch.save(model2.state_dict(), args.output_model)
        print(f"\n💾 最终模型已保存到: {args.output_model}")
    # ========================================================
    # ========================================================

    print("\n生成输出:")
    os.environ["MPO_EVAL_PATH"] = "mpo"
    func2_outputs = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        output = generate_text(model2, tokenizer, prompt, max_new_tokens=30)
        func2_outputs.append(output)
        print(f"   Output: {output}")

    print("\nPPL评估...")
    t0 = time.time()
    ppl2 = eval_ppl(model2, tokenizer, max_tokens=50000)
    print(f"函数2(调整版) PPL: {ppl2:.2f} (ΔPPL: {ppl2 - orig_ppl:+.2f})")
    print(f"评估耗时: {time.time() - t0:.1f}s")

    results["func2_adjusted"] = {"ppl": ppl2, "stats": stats2, "outputs": func2_outputs, "time": compress_time2}

    del model2
    torch.cuda.empty_cache()

    # ==========================================
    # 测试3: 满秩MPO (无截断)
    # ==========================================
    print_section("【步骤6】测试3: 满秩MPO (无截断, bond_dim=999999)")

    print("重新加载模型...")
    model3 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )

    print("\n开始压缩 (满秩无截断)...")
    t0 = time.time()
    model3 = compress_full_rank(model3, cfg)
    compress_time3 = time.time() - t0

    stats3 = count_params(model3)
    print(f"\n压缩耗时: {compress_time3:.1f}s")
    print(f"参数量统计:")
    print(f"  总参数量: {stats3['total'] / 1e6:.1f}M ({stats3['total'] / orig_stats['total']:.1%})")
    print(f"  MPO层数: {stats3['mpo_layers']}")
    print(f"  MPO压缩率: {stats3['mpo_params'] / stats3['dense_equivalent']:.1%}")

    print("\n生成输出:")
    os.environ["MPO_EVAL_PATH"] = "dense"
    fullrank_outputs = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        output = generate_text(model3, tokenizer, prompt, max_new_tokens=30)
        fullrank_outputs.append(output)
        print(f"   Output: {output}")

    print("\nPPL评估...")
    t0 = time.time()
    ppl3 = eval_ppl(model3, tokenizer, max_tokens=50000)
    print(f"满秩MPO PPL: {ppl3:.2f} (ΔPPL: {ppl3 - orig_ppl:+.2f})")
    print(f"评估耗时: {time.time() - t0:.1f}s")

    results["full_rank"] = {"ppl": ppl3, "stats": stats3, "outputs": fullrank_outputs, "time": compress_time3}

    del model3
    torch.cuda.empty_cache()

    # ==========================================
    # 完整对比总结
    # ==========================================
    print_section("【步骤7】完整对比总结", "=")

    # 1. 参数量对比
    print("\n📊 1. 参数量对比")
    print("-" * 70)
    print(f"{'模型':<30} {'总参数量':>12} {'vs原始':>10} {'MPO压缩率':>12}")
    print("-" * 70)
    print(f"{'原始模型':<30} {orig_stats['total']/1e6:>10.1f}M {'100.0%':>10} {'-':>12}")
    print(f"{'函数1 (有截断, 无s_vector)':<30} {stats1['total']/1e6:>10.1f}M {stats1['total']/orig_stats['total']:>9.1%} {stats1['mpo_params']/stats1['dense_equivalent']:>11.1%}")
    print(f"{'函数2 (有截断, 有s_vector)':<30} {stats2['total']/1e6:>10.1f}M {stats2['total']/orig_stats['total']:>9.1%} {stats2['mpo_params']/stats2['dense_equivalent']:>11.1%}")
    print(f"{'满秩MPO (无截断)':<30} {stats3['total']/1e6:>10.1f}M {stats3['total']/orig_stats['total']:>9.1%} {stats3['mpo_params']/stats3['dense_equivalent']:>11.1%}")
    print("-" * 70)

    # 2. PPL对比
    print("\n📈 2. PPL对比")
    print("-" * 70)
    print(f"{'模型':<30} {'PPL':>10} {'ΔPPL':>12} {'相对变化':>12}")
    print("-" * 70)
    print(f"{'原始模型':<30} {orig_ppl:>10.2f} {'-':>12} {'-':>12}")
    print(f"{'函数1 (有截断, 无s_vector)':<30} {ppl1:>10.2f} {ppl1-orig_ppl:>+11.2f} {(ppl1-orig_ppl)/orig_ppl*100:>+10.1f}%")
    print(f"{'函数2 (有截断, 有s_vector)':<30} {ppl2:>10.2f} {ppl2-orig_ppl:>+11.2f} {(ppl2-orig_ppl)/orig_ppl*100:>+10.1f}%")
    print(f"{'满秩MPO (无截断)':<30} {ppl3:>10.2f} {ppl3-orig_ppl:>+11.2f} {(ppl3-orig_ppl)/orig_ppl*100:>+10.1f}%")
    print("-" * 70)

    # 3. 压缩耗时对比
    print("\n⏱️  3. 压缩耗时对比")
    print("-" * 70)
    print(f"{'模型':<30} {'耗时':>10}")
    print("-" * 70)
    print(f"{'函数1 (有截断, 无s_vector)':<30} {compress_time1:>9.1f}s")
    print(f"{'函数2 (有截断, 有s_vector)':<30} {compress_time2:>9.1f}s")
    print(f"{'满秩MPO (无截断)':<30} {compress_time3:>9.1f}s")
    print("-" * 70)

    # 4. 生成文本对比
    print("\n📝 4. 生成文本对比")
    print("-" * 70)
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {i}: '{prompt}'")
        print(f"  原始:     {orig_outputs[i-1][:80]}...")
        print(f"  函数1:    {func1_outputs[i-1][:80]}...")
        print(f"  函数2:    {func2_outputs[i-1][:80]}...")
        print(f"  满秩MPO:  {fullrank_outputs[i-1][:80]}...")

        # 检查满秩是否与原始相同
        if fullrank_outputs[i-1] == orig_outputs[i-1]:
            print("  → 满秩MPO与原始输出相同 ✓")
        else:
            print("  → 满秩MPO与原始输出有差异")

    # 5. 关键发现
    print("\n🔍 5. 关键发现")
    print("-" * 70)

    # 满秩效果分析
    ppl_diff_full = abs(ppl3 - orig_ppl)
    if ppl_diff_full < 0.5:
        print("✅ 满秩MPO几乎无精度损失！PPL差异 < 0.5")
    elif ppl_diff_full < 5:
        print(f"⚠️  满秩MPO有轻微精度损失，PPL差异: {ppl_diff_full:.2f}")
    else:
        print(f"❌ 满秩MPO精度损失较大，PPL差异: {ppl_diff_full:.2f}")

    # 参数量分析
    if stats3['total'] > orig_stats['total']:
        overhead = (stats3['total'] - orig_stats['total']) / orig_stats['total'] * 100
        print(f"📊 满秩MPO参数量增加 {overhead:.1f}% (MPO结构开销)")

    # SVD截断影响
    print(f"\n📊 SVD截断影响分析:")
    print(f"  有截断(函数1): PPL从 {orig_ppl:.2f} → {ppl1:.2f} (+{(ppl1-orig_ppl)/orig_ppl*100:.0f}%)")
    print(f"  有截断(函数2): PPL从 {orig_ppl:.2f} → {ppl2:.2f} (+{(ppl2-orig_ppl)/orig_ppl*100:.0f}%)")
    print(f"  无截断(满秩):  PPL从 {orig_ppl:.2f} → {ppl3:.2f} (+{(ppl3-orig_ppl)/orig_ppl*100:.0f}%)")

    print("\n💡 结论:")
    if ppl3 < min(ppl1, ppl2):
        print("  - 满秩MPO (无截断) 性能最好，证明SVD截断是主要损失来源")
    if stats3['total'] > stats1['total']:
        print(f"  - 满秩MPO参数量 ({stats3['total']/1e6:.1f}M) > 有截断 ({stats1['total']/1e6:.1f}M)")
        print("  - 需要在压缩率和精度之间权衡")

    print_section("测试完成", "=")


if __name__ == "__main__":
    main()
