#!/usr/bin/env python3
"""
MPO + TT-Embedding 端到端压缩与微调脚本
只保留最高效的 factor_linear_mpo_custom (有截断, 有 s_vector)
并引入 Tensor Train (TT) 对 Embedding 层进行压缩。
"""

import os
import sys
import time
from pathlib import Path
import torch.nn.functional as F
import math

# 强制使用 4 号 GPU
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
    parser = argparse.ArgumentParser(description="MPO + TT-Embedding 压缩测试脚本")
    parser.add_argument("--model", type=str, default="tinyllama", choices=["tinyllama", "llama-7b"])
    parser.add_argument("--num_cores", type=int, default=3, help="MPO的长度")
    parser.add_argument("--boundary", type=str, default="open", help="边界条件")
    parser.add_argument("--target_ratio", type=float, default=0.20, help="MPO 中层目标压缩率")
    # [新增]
    parser.add_argument("--deep_ratio", type=float, default=None, help="MPO 深层目标压缩率 (默认同 target_ratio)")
    parser.add_argument("--lora_rank", type=int, default=16, help="Res-MPO 中残差辅助矩阵的秩")
    parser.add_argument("--use_s_vector", action="store_true", help="使用 ASVD 保护异常激活值")
    
    # [新增] TT-Embedding 控制开关
    parser.add_argument("--compress_embedding", action="store_true", help="是否对 Embedding 层进行 TT 压缩")
    parser.add_argument("--tt_rank", type=int, default=64, help="TT-Embedding 的内部键维 (Rank)")

    # 愈合训练控制开关
    parser.add_argument("--do_healing", action="store_true", help="开启愈合微调")
    parser.add_argument("--data_mode", type=str, default="wiki_then_chat", 
                        choices=["wiki", "chat", "wiki_then_chat", "mixed"], 
                        help="愈合模式: 只用wiki / 只用chat / 先wiki后chat")
    parser.add_argument("--healing_epochs", type=int, default=1)
    # 注意：这里的全局 LR 将在 healing.py 中被拆分
    parser.add_argument("--healing_lr", type=float, default=5e-5) 
    parser.add_argument("--save_every_n_steps", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="./healing_checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_model", type=str, default=None)
    return parser.parse_args()

# ==========================================
# 核心新增：TT-Embedding 类与分解逻辑
# ==========================================

class TTEmbedding(nn.Module):
    """
    Tensor Train Embedding (TT-Embedding) 实现

    核心思路：
    ---------
    传统的 Embedding 层是一个稠密矩阵 W ∈ ℝ^(V×H)，其中 V 是词表大小，H 是隐藏维度。
    我们可以将其视为一个 Matrix Product Operator (MPO)：

        W[i, j] = Σ_{r1,r2} A[0][i0,j0,r1] · A[1][i1,j1,r1,r2] · A[2][i2,j2,r2]

    其中：
    - i = (i0, i1, i2) 是词表索引的 3D 分解（例如 32000 = 32×10×100）
    - j = (j0, j1, j2) 是隐藏维度的 3D 分解（例如 2048 = 8×16×16）

    TT-Embedding 的关键洞察：
    -------------------------
    当给定一个具体的 token ID 时，我们实际上固定了词表维度的具体值 (i0, i1, i2)。
    此时 MPO 退化为一个 Matrix Product State (MPS)：

        v[j] = Σ_{r1,r2} A[0][i0,j0,r1] · A[1][i1,j1,r1,r2] · A[2][i2,j2,r2]

    即：通过索引从每个 core 中抽取对应的切片，然后通过张量收缩合并 bond 维度，
    最终展平得到该 token 的 hidden vector。

    参数说明：
    ---------
    - v_factors: 词表维度的分解，如 [32, 10, 100] 表示 32000 = 32×10×100
    - h_factors: 隐藏维度的分解，如 [8, 16, 16] 表示 2048 = 8×16×16
    - cores: 3个张量核的参数列表，形状分别为：
        - cores[0]: (v0, h0, r1)  即 (32, 8, r1)
        - cores[1]: (v1, r1, h1, r2) 即 (10, r1, 16, r2)
        - cores[2]: (v2, r2, h2) 即 (100, r2, 16)
    """

    def __init__(self, v_factors, h_factors, cores, padding_idx=None):
        super().__init__()
        self.v = v_factors
        self.h = h_factors
        # cores 是一个包含 3 个张量核的 nn.ParameterList
        self.cores = cores
        # ✅ 兼容 HuggingFace 接口
        self.num_embeddings = v_factors[0] * v_factors[1] * v_factors[2]
        self.embedding_dim = h_factors[0] * h_factors[1] * h_factors[2]

        # 保存 padding_idx
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        """
        前向传播：将 token ID 映射为 hidden vector

        步骤：
        1. 将 1D token ID 分解为 3D 坐标 (i1, i2, i3)
        2. 根据坐标从每个 core 中动态抽取切片
        3. 通过 einsum 收缩 bond 维度，合并三个 cores
        4. 展平得到最终的 hidden_size 向量
        """
        # 1. 将 1D 的 Token ID 拆解为 3D 坐标
        #    例如：token_id = 12345 → (i1, i2, i3) = (3, 8, 45)
        i1 = input_ids // (self.v[1] * self.v[2])
        rem = input_ids % (self.v[1] * self.v[2])
        i2 = rem // self.v[2]
        i3 = rem % self.v[2]

        # 2. 动态抽取切片
        #    关键操作：通过 token ID 索引，从 MPO 中"选择"出对应的一行
        #    此时 MPO 退化为 MPS，physical dim 被固定
        c1 = self.cores[0][i1]  # [batch, seq_len, 1, h1, r1]
        c2 = self.cores[1][i2]  # [batch, seq_len, r1, h2, r2]
        c3 = self.cores[2][i3]  # [batch, seq_len, r2, h3, 1]

        # 3. 实时张量收缩 (Tensor Contraction)
        #    将 MPS 的三个 cores 沿着 bond 维度 (r1, r2) 收缩合并
        #    同时合并 physical 维度 (h1, h2, h3) 得到最终的 hidden vector
        # x: [batch, seq_len, 1, h1, h2, r2]
        x = torch.einsum('b s i h r, b s r j p -> b s i h j p', c1, c2)
        x = x.squeeze(2) # 移除维度 1 -> [batch, seq_len, h1, h2, r2]

        # out: [batch, seq_len, h1, h2, h3, 1]
        out = torch.einsum('b s h j r, b s r k p -> b s h j k p', x, c3)

        # 4. 展平为最终的 hidden_size
        #    (h1, h2, h3) → H，即 8×16×16 = 2048
        out = out.view(input_ids.shape[0], input_ids.shape[1], -1)

        # ====================================================
        # 💡 5. 核心修复：强行阻断 padding_idx 的梯度回流！
        # ====================================================
        if self.padding_idx is not None:
            # 生成一个 mask: 不是 pad 的地方是 1.0，是 pad 的地方是 0.0
            # [batch, seq_len, 1] 形状，方便与 out 进行广播相乘
            mask = (input_ids != self.padding_idx).unsqueeze(-1).to(out.dtype)
            
            # 乘以 mask！
            # pad 位置的向量变成全 0，计算图在这里被物理切断，梯度绝对不会流回 cores
            out = out * mask
            
        return out

# ==========================================
# 核心新增：Res-MPO 包装器 (带误差截断初始化)
# ==========================================


class ResMPOWrapper(nn.Module):
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig, skip_svd=False):
        super().__init__()
        self.mpo = mpo_module
        self.r = lora_rank
        if skip_svd:
            # 只声明参数形状，不做 SVD（反正马上会被 load_state_dict 覆盖）
            target_device = W_orig.device
            target_dtype = W_orig.dtype
            self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, device=target_device, dtype=target_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=target_device, dtype=target_dtype))
            return
    

        # ✅ 记录最终目标设备和类型，但所有计算在 CPU float32 上做
        target_device = W_orig.device
        target_dtype = W_orig.dtype

        print(f"      [Res-MPO] 正在通过 SVD(ΔW) 初始化 LoRA 矩阵 (r={self.r})...")
        with torch.no_grad():
            # ✅ 把 MPO 临时搬到 CPU 做提取
            mpo_cpu = self.mpo.cpu().float()
            
            # ✅ 强制把 s_vector 也拉到 CPU（防止它不是 buffer 而是普通属性）
            if hasattr(mpo_cpu, 's_vector') and mpo_cpu.s_vector is not None:
                mpo_cpu.s_vector = mpo_cpu.s_vector.cpu().float()
                
            bias_backup = None
            if hasattr(mpo_cpu, 'bias') and mpo_cpu.bias is not None:
                bias_backup = mpo_cpu.bias.data.clone()
                mpo_cpu.bias.data.zero_()

            eye = torch.eye(in_features, dtype=torch.float32)  # CPU
            W_mpo = mpo_cpu(eye).T                               # CPU

            if bias_backup is not None:
                mpo_cpu.bias.data.copy_(bias_backup)

            # ✅ W_orig 也拉到 CPU
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

        # ✅ 把 MPO 搬回 GPU
        # ✅ 搬回 GPU 后恢复 s_vector
        self.mpo = self.mpo.to(device=target_device, dtype=target_dtype)
        if hasattr(self.mpo, 's_vector') and self.mpo.s_vector is not None:
            self.mpo.s_vector = self.mpo.s_vector.to(device=target_device, dtype=target_dtype)
        self.lora_A.data = self.lora_A.data.to(target_device)
        self.lora_B.data = self.lora_B.data.to(target_device)

    def forward(self, x):
        # MPO 路径
        mpo_out = self.mpo(x)
        # LoRA 残差路径: x @ A^T @ B^T
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return mpo_out + lora_out


def factorize_vocab(V, num_factors=3, max_pad_ratio=0.01):
    best = None
    best_prod = float('inf')
    best_spread = float('inf')  # ← 新增：衡量均衡性
    
    upper = int(math.isqrt(V)) + 1
    for f1 in range(2, upper):
        for f2 in range(f1, upper):
            f3 = math.ceil(V / (f1 * f2))
            if f3 < 2:
                continue
            prod = f1 * f2 * f3
            if prod < V or (prod - V) / V > max_pad_ratio:
                continue
            spread = max(f1, f2, f3) - min(f1, f2, f3)  # ← 越小越均衡
            if prod < best_prod or (prod == best_prod and spread < best_spread):
                best = sorted([f1, f2, f3])
                best_prod = prod
                best_spread = spread
    
    if best is None:
        if max_pad_ratio >= 0.10:  # ← 终止条件
            raise ValueError(f"无法将 {V} 分解为 {num_factors} 个因子 (max_pad_ratio={max_pad_ratio})")
        return factorize_vocab(V, num_factors, max_pad_ratio=max_pad_ratio * 2)
    return best_prod, best

@torch.no_grad()
def replace_embedding_with_tt(model, cfg, tt_rank=64, skip_svd=False, loaded_weights=None):
    # ... (前面的原注释省略) ...
    """
    使用 TT-SVD 将 Embedding 层替换为 Tensor Train 结构 (带自动补齐机制)

    TT-SVD 分解原理：
    ----------------
    原始 Embedding 权重 W ∈ ℝ^(V×H) 可以重塑为 6 维张量：
        W_tensor ∈ ℝ^(v0×v1×v2×h0×h1×h2)

    其中：
    - V = v0 × v1 × v2  (词表分解)
    - H = h0 × h1 × h2  (隐藏维度分解)

    通过连续 SVD 分解，将 W_tensor 分解为 3 个 TT cores：
        W[i0,i1,i2,j0,j1,j2] ≈ Σ_{r1,r2} G1[i0,j0,r1] · G2[i1,j1,r1,r2] · G3[i2,j2,r2]

    关键步骤：
    1. SVD #1: W1 ∈ ℝ^((v0·h0)×(v1·v2·h1·h2)) → U1·S1·V1
       取前 r1 个奇异值，得到 core1: G1 = reshape(U1[:, :r1]) → (v0, h0, r1)

    2. SVD #2: W2 = diag(S1)·V1 ∈ ℝ^((r1·v1·h1)×(v2·h2)) → U2·S2·V2
       取前 r2 个奇异值，得到 core2: G2 = reshape(U2[:, :r2]) → (v1, h1, r1, r2)

    3. Core3: G3 = reshape(V2[:r2, :]) → (v2, h2, r2)

    压缩效果：
    - 原始参数: V × H (如 32000 × 2048 ≈ 65.5M)
    - TT 参数: v0·h0·r1 + v1·h1·r1·r2 + v2·h2·r2 (约 2-5M，取决于 rank)
    - 典型压缩率: 10-30x
    """
    embed_layer = model.model.embed_tokens
    W = embed_layer.weight.detach().clone().float()  # 在 GPU 上
    V, H = W.shape
    device, dtype = embed_layer.weight.device, embed_layer.weight.dtype 

    # 1. 通用 Vocab Size 均衡分解与补齐
    target_V, v_factors = factorize_vocab(V, num_factors=3)
        
    pad_len = target_V - V
    if pad_len > 0:
        print(f"⚠️ 词表大小为 {V}，自动均衡分解为 {v_factors}。")
        print(f"⚠️ 正在补齐 {pad_len} 个 Dummy Tokens 至 {target_V}...")
        W = F.pad(W, (0, 0, 0, pad_len), "constant", 0)
    else:
        print(f"✅ 词表大小 {V} 无需补齐，直接分解为 {v_factors}。")

    # 2. 自动处理 Hidden Size
    if H == 2048:
        h_factors = [8, 16, 16]   
    elif H == 4096:
        h_factors = [16, 16, 16]  
    else:
        raise ValueError(f"不支持的 Hidden Size: {H}")

    # ==========================================================
    # 💡 核心修复：跳过 SVD，直接通过查字典拿到真实形状搭骨架
    # ==========================================================
    if skip_svd and loaded_weights is not None:
        print("\n⚡ [快速恢复模式] TT-Embedding 查字典跳过 SVD，直接搭建空壳...")
        
        # 在 Llama 的 state_dict 中，embed_tokens 的真实前缀
        prefix = "model.embed_tokens.cores."
        if f"{prefix}0" in loaded_weights:
            s1 = loaded_weights[f"{prefix}0"].shape
            s2 = loaded_weights[f"{prefix}1"].shape
            s3 = loaded_weights[f"{prefix}2"].shape
            
            # 用真实形状造零张量
            core1 = torch.zeros(s1, device=device, dtype=dtype)
            core2 = torch.zeros(s2, device=device, dtype=dtype)
            core3 = torch.zeros(s3, device=device, dtype=dtype)
            
            # 释放用不到的 W
            del W
        else:
            print("⚠️ 字典中未找到 TT-Embedding，回退到完整 SVD 计算。")
            skip_svd = False

    # 如果没有 skip_svd，则老老实实做 SVD
    if not skip_svd:
        print(f"\n🔄 正在执行 TT-SVD 压缩 Embedding...")
        
        W_tensor = W.view(v_factors[0], v_factors[1], v_factors[2], h_factors[0], h_factors[1], h_factors[2])
        W_tensor = W_tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
        del W

        # Core 1 SVD
        W1 = W_tensor.view(v_factors[0] * h_factors[0], -1)
        U1, S1, V1 = torch.linalg.svd(W1, full_matrices=False)
        r1 = min(tt_rank, S1.shape[0])
        core1 = U1[:, :r1].view(v_factors[0], h_factors[0], r1).unsqueeze(1)
        del W1, U1

        # Core 2 & 3 SVD
        W2 = (torch.diag(S1[:r1]) @ V1[:r1, :]).view(r1 * v_factors[1] * h_factors[1], -1)
        del S1, V1

        U2, S2, V2 = torch.linalg.svd(W2, full_matrices=False)
        r2 = min(tt_rank, S2.shape[0])

        core2 = U2[:, :r2].view(r1, v_factors[1], h_factors[1], r2).permute(1, 0, 2, 3)
        core3 = (torch.diag(S2[:r2]) @ V2[:r2, :]).view(r2, v_factors[2], h_factors[2], 1).permute(1, 0, 2, 3)

        del W2, U2, S2, V2, W_tensor

    # ==========================================================
    # 包装并替换层
    # ==========================================================
    cores = nn.ParameterList([
        nn.Parameter(core1.to(device=device, dtype=dtype)),
        nn.Parameter(core2.to(device=device, dtype=dtype)),
        nn.Parameter(core3.to(device=device, dtype=dtype))
    ])
    
    original_padding_idx = embed_layer.padding_idx 
    tt_embed = TTEmbedding(v_factors, h_factors, cores, padding_idx=original_padding_idx)
    model.model.embed_tokens = tt_embed
    print("✅ TT-Embedding 替换/搭建成功！")
    
    return model

# ==========================================
# 辅助函数 (原版保留)
# ==========================================
def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=50000):
    model.eval() # 💡 新增：确保模型处于推理状态
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
    total = sum(p.numel() for p in model.parameters())
    tt_embed_params = 0
    
    for name, mod in model.named_modules():
        if isinstance(mod, TTEmbedding):
            tt_embed_params = sum(p.numel() for p in mod.parameters())
            
    return {
        "total": total,
        "tt_embed_params": tt_embed_params
    }

# ==========================================
# 新增辅助函数：解析计算 MPO core 形状
# ==========================================

def compute_mpo_core_shapes(out_fac, in_fac, bond_dim, num_cores):
    """
    不做 SVD，直接根据维度因子和 bond_dim 解析地算出每个 core 的形状。
    用于从 checkpoint 恢复时快速构建空壳 MPOLinear，跳过昂贵的 SVD 分解。
    
    原理：factor_linear_mpo_custom 第 k 步的 SVD 截断秩为
        r_k = min(bond_dim, min(rows_k, cols_k))
    其中 rows_k = r_{k-1} * o_k * i_k,
         cols_k = ∏_{j=k+1}^{K-1} (o_j * i_j)
    这些量只依赖维度因子，不需要做任何矩阵运算。
    """
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
# 修改后的压缩主函数
# ==========================================

@torch.no_grad()
def compress_with_function2(model, cfg, activation_scales=None, skip_svd=False, loaded_weights=None):
    """
    MPO + Res-LoRA 压缩主函数。

    当 skip_svd=True 时，跳过所有 SVD 分解，仅构建形状正确的空壳模块，
    配合 load_state_dict 从 checkpoint 恢复，将恢复时间从 O(分钟) 降到 O(秒)。
    """
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

            # 找到这段 skip_svd 的判断逻辑，替换成下面这样：
            if skip_svd:
                # ⚡ 快速路径：优先查字典获取真实形状，查不到再算理论最大值
                if loaded_weights is not None:
                    # 尝试从字典里直接读取真实保存的 Shape
                    core_shapes = []
                    for c_idx in range(num_cores):
                        # 拼凑出这个 core 在字典里的真实名字
                        weight_name = f"model.layers.{idx}.mlp.{fname}.mpo.cores.{c_idx}"
                        if weight_name in loaded_weights:
                            core_shapes.append(loaded_weights[weight_name].shape)
                        else:
                            # 万一没存这个层，回退到理论算法
                            print(f"⚠️ 未找到 {weight_name}，回退到理论形状")
                            core_shapes = compute_mpo_core_shapes(out_fac, in_fac, chi_ffn, num_cores)
                            break
                else:
                    # 如果压根没传字典，走纯理论算法
                    core_shapes = compute_mpo_core_shapes(out_fac, in_fac, chi_ffn, num_cores)
                
                # 用拿到的真实形状构建 dummy 张量
                dummy_cores = [torch.zeros(s, device=device, dtype=dtype0) for s in core_shapes]
                mpo = MPOLinear(in_f, out_f, dummy_cores, s_vector=s_vector, boundary="open")
            else:
                # 🐢 完整路径：CPU 上做 SVD 分解
                W = lin.weight.detach().clone().cpu().float()
                s_vector_cpu = s_vector.cpu().float() if s_vector is not None else None

                cores_list = factor_linear_mpo_custom(
                    weight=W, bond_dim=chi_ffn, num_cores=num_cores,
                    out_fac=out_fac, in_fac=in_fac,
                    s_vector=s_vector_cpu,
                    boundary="open",
                    adaptive=True, energy_threshold=0.99
                )

                cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]
                mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vector, boundary="open")

                if getattr(lin, "bias", None) is not None:
                    with torch.no_grad():
                        mpo.bias.copy_(lin.bias.data)

                del W, cores_list, cleaned_cores

            lora_rank = cfg.get("lora_rank", 16)

            if skip_svd:
                # skip_svd 时 ResMPOWrapper 只需要知道 device/dtype，不需要真正的 W_orig
                # 传 lin.weight 进去（不 clone，省显存），反正只读 .device 和 .dtype
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
            # ⚡ 快速路径：优先查字典获取真实形状，查不到再算理论最大值
            if loaded_weights is not None:
                core_shapes = []
                for c_idx in range(num_cores):
                    # LM Head 在模型 state_dict 里的真实前缀是没有 model. 的
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
            # 🐢 完整路径：CPU/GPU 上做 SVD 分解
            W_head = lm_head.weight.detach().clone().cpu().float()

            cores_list_head = factor_linear_mpo_custom(
                weight=W_head, bond_dim=chi_head, num_cores=num_cores,
                out_fac=out_fac_head, in_fac=in_fac_head, boundary="open",adaptive=True, energy_threshold=0.999
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

def main():
    # 程序开始计时
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

    # 如果命令行没有指定 deep_ratio，就让它强制等于 target_ratio
    deep_ratio = args.deep_ratio if args.deep_ratio is not None else args.target_ratio

    cfg = {
        "num_cores": args.num_cores,
        "boundary": args.boundary,
        "freeze_blocks": freeze_blocks,
        "mid_blocks": mid_blocks,
        "target_ratio": args.target_ratio,
        "deep_ratio": deep_ratio,  # 💡 修复 5：解绑硬编码！
        "skip_mlp": "down_proj",
        "lora_rank": args.lora_rank,
        "head_ratio": 0.5 # 刚才修的那个给 LM Head 的宽容压缩率
    }

    print("\n" + "="*70)
    print("MPO FC + TT-Embedding 纯净压缩微调架构")
    print("="*70)

    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ====================================================
    # 💡 修复：强行注入 Chat Template (使用主流的 ChatML 或 Llama 格式)
    # ====================================================
    if tokenizer.chat_template is None:
        print("⚠️ 未检测到默认对话模板，正在强行注入默认模板...")
        # 这是一个非常通用且清爽的模板 (类似 TinyLlama / ChatML 的风格)
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

    # 2. 提取 ASVD (按需)
    if args.use_s_vector:
        print("\n提取 activation scales...")
        activation_scales_dict = get_activation_scales(model, tokenizer, num_samples=128, max_len=512)
    else:
        activation_scales_dict = None

    orig_stats = count_params(model)

    # ====================================================
    # 💡 新增：第一次称重 (手术前) 
    # 遍历所有名字里带 "mlp" 的参数，算出原始的 FC 层总参数量
    # ====================================================
    orig_mlp_params = sum(p.numel() for name, p in model.named_parameters() if "mlp" in name)

    # 3. 执行 TT-Embedding 压缩
    if args.compress_embedding:
        model = replace_embedding_with_tt(model, cfg, tt_rank=args.tt_rank)

    # 4. 执行 MPO FC 压缩
    # 定义存档路径
    surgery_checkpoint = (
    f"compressed_{args.model}"
    f"_cores{args.num_cores}"
    f"_r{args.lora_rank}"
    f"_tr{args.target_ratio}"
    f"_dr{deep_ratio}"
    f"_sv{int(args.use_s_vector)}"
    f"_emb{int(args.compress_embedding)}_ttR{args.tt_rank}"
    f".pt"
    )

    # ====================================================
    # 💡 核心修复：把读取存档提到最前面！
    # ====================================================
    trainable_weights = None
    is_resume = os.path.exists(surgery_checkpoint)
    
    if is_resume:
        print(f"\n⚠️ 检测到手术存档 {surgery_checkpoint}！")
        print("   📂 正在提前读取存档，用于全局空壳骨架搭建...")
        ckpt = torch.load(surgery_checkpoint, map_location="cpu")
        trainable_weights = ckpt.get("trainable_state_dict", ckpt)

    # 3. 执行 TT-Embedding 压缩 (传入 skip 标志和权重)
    if args.compress_embedding:
        model = replace_embedding_with_tt(
            model, cfg, tt_rank=args.tt_rank, 
            skip_svd=is_resume,                 # 如果是恢复模式，跳过 SVD
            loaded_weights=trainable_weights    # 递上字典让它抄形状
        )

    # 4. 执行 MPO FC 压缩
    if is_resume:
        # 直接用极速模式搭完剩下的所有骨架
        model = compress_with_function2(
            model, cfg, activation_scales=activation_scales_dict, 
            skip_svd=True, loaded_weights=trainable_weights
        )
        
        # 💡 骨架全部就绪，一次性严丝合缝地灌入所有灵魂！
        result = model.load_state_dict(trainable_weights, strict=False)
        if result.missing_keys:
            print(f"   ⚠️ [警告] 缺失 {len(result.missing_keys)} 个权重! 示例: {result.missing_keys[:3]}")
        if result.unexpected_keys:
            print(f"   ⚠️ [警告] 多出 {len(result.unexpected_keys)} 个未知权重! 示例: {result.unexpected_keys[:3]}")
        else:
            print("   ✅ 所有存档权重 (含 TT-Embedding & FC) 完美匹配！")
    else:
        # 如果是首次运行，走完整 SVD 流程
        model = compress_with_function2(model, cfg, activation_scales=activation_scales_dict)
        torch.save(model.state_dict(), surgery_checkpoint)
    
    stats = count_params(model)
    print(f"\n压缩后总参数量: {stats['total'] / 1e6:.1f}M ({stats['total'] / orig_stats['total']:.1%})")
    if args.compress_embedding:
        print(f"TT-Embedding 参数量: {stats['tt_embed_params'] / 1e6:.2f}M (极大压缩了词表)")

    # ====================================================
    # 💡 新增：第二次称重 (手术后)
    # 再次统计带 "mlp" 的参数。此时它已经自动包含了：
    # 压缩的 MPO 核 + SVD 初始化的 LoRA + 被跳过未压缩的 Down 矩阵！
    # ====================================================
    new_mlp_params = sum(p.numel() for name, p in model.named_parameters() if "mlp" in name)

    # 5. 执行 Healing (支持两段式蒸馏)
    if args.do_healing:
        print(f"\n🚀 启动端到端联合蒸馏...")

        print("\n📦 正在获取 Student 模型的设备拓扑图...")
        # 获取 Student 最原始的物理层级分布字典
        student_device_map = model.hf_device_map

        print("📦 正在强制按照 Student 的拓扑图加载 Teacher 模型...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=student_device_map  # 👈 核心修复：强行对齐两者的器官位置！
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        os.environ["MPO_EVAL_PATH"] = "mpo"
        os.environ["MPO_TRAIN_PATH"] = "mpo"

        if args.data_mode in ["wiki", "wiki_then_chat"]:
            # 第一阶段：大水漫灌，恢复张量物理连接 (LR稍大)
            model = train_healing(
                student_model=model,
                tokenizer=tokenizer,
                teacher_model=teacher_model, # 传入刚加载的 Teacher
                dataset_name="wiki",
                epochs=args.healing_epochs,
                batch_size=1,  # 必须是1，双模型极吃显存
                accum_steps=8,  # 拉高累计步数
                lr=args.healing_lr,
                seq_len=256,
                save_every_n_steps=args.save_every_n_steps,
                checkpoint_dir=args.checkpoint_dir + "_wiki",
                max_update_steps=2000
            )

        # 💡 在命令行参数里支持 "mixed"
        if args.data_mode in ["chat", "wiki_then_chat", "mixed"]:
            # 第二阶段：混合神药规训，注入灵魂与逻辑
            print("\n🔄 切换到混合数据集 (Mixed) MPO 路径训练...")
            
            # 💡 这里写死你最好的那个 1200 步的路径
            my_resume_path = args.resume_from if args.resume_from else "/home/roots/xiaoshi/LLM-main/MPO_Compression/healing_checkpoints_mixed/checkpoint_upd_1600.pt"
            os.environ["MPO_TRAIN_PATH"] = "mpo"
            
            model = train_healing(
                student_model=model,
                tokenizer=tokenizer,
                teacher_model=teacher_model, 
                
                dataset_name="mixed",  # 💡 1. 改为 mixed！
                
                epochs=args.healing_epochs,
                batch_size=4,
                accum_steps=2,
                
                lr=args.healing_lr,
                
                seq_len=1024,  # 💡 3. 如果一跑就 OOM，改回 512
                
                save_every_n_steps=args.save_every_n_steps,
                checkpoint_dir=args.checkpoint_dir + "_mixed", # 💡 存到新文件夹，别和之前的混了
                
                max_update_steps=1200,  
                
                resume_from_checkpoint=my_resume_path 
            )

    # ------------------ compress_embedding.py 追加在 main() 末尾 ------------------
    if args.output_model:
        torch.save(model.state_dict(), args.output_model)
        print(f"\n💾 模型已保存: {args.output_model}")



    # ==========================================
    # 6. 终极战报：PPL评估与全局压缩率
    # ==========================================
    stats_final = count_params(model)
    
    # 原始的 Embedding 参数量 (假设是 TinyLlama 32000 * 2048)
    # 如果词表被 pad 到了 32010，我们依然按原始来算账
    orig_embed_size = model.config.vocab_size * model.config.hidden_size
    
    print("\n" + "="*70)
    print("🏆 终极评估战报 (End-to-End Tensorization)")
    print("="*70)

    print(f"📊 【FC 层 (MLP Block)】")
    print(f"   原 Dense 参数: {orig_mlp_params / 1e6:.1f} M")
    print(f"   现 混合 参数:  {new_mlp_params / 1e6:.1f} M (含 MPO + LoRA + 跳过的层)")
    print(f"   真实压缩率:    {new_mlp_params / orig_mlp_params:.1%}")  # 👈 最诚实的数据！

    if args.compress_embedding:
        print(f"\n📊 【词表层 (TT-Embedding)】")
        print(f"   原 词表 参数:  {orig_embed_size / 1e6:.1f} M")
        print(f"   现 TT 参数:    {stats_final['tt_embed_params'] / 1e6:.1f} M")
        print(f"   词表 压缩率:   {stats_final['tt_embed_params'] / orig_embed_size:.1%}")

    print(f"\n📊 【全局总计】")
    print(f"   原模型总参数:  {orig_stats['total'] / 1e6:.1f} M")
    print(f"   现模型总参数:  {stats_final['total'] / 1e6:.1f} M")
    print(f"   全局保留率:    {stats_final['total'] / orig_stats['total']:.1%}")

    # 评估 PPL
    print("\n⏳ 正在评估最终模型的 PPL (Wikitext-2)...")
    os.environ["MPO_EVAL_PATH"] = "mpo" # 确保前向传播走你的张量算子
    t0 = time.time()
    final_ppl = eval_ppl(model, tokenizer, max_tokens=50000)
    print(f"✅ 最终模型 PPL: {final_ppl:.2f}")
    print(f"   (评估耗时: {time.time() - t0:.1f}s)")
    print("="*70)

    # 程序结束计时并输出总运行时间
    total_time = time.time() - program_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n⏱️ 程序总运行时间: {hours}小时 {minutes}分钟 {seconds:.1f}秒")
    print(f"   (Total: {total_time:.1f}s)")
    print("="*70)

if __name__ == "__main__":
    main()
    






