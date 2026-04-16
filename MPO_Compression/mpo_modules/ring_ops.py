#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Modules - Ring (Periodic Boundary) 操作

包含 MPO ring 分解、ring 核收缩、ring 秩信息查询等操作。
Ring MPO 与 chain (open boundary) TT 的区别在于首尾核共享一个周期性虚 bond，
即 core 形状为 [r₀, o₁, i₁, r₁], ..., [r_{K-1}, o_K, i_K, r₀]，
还原时对 r₀ 维做 trace 而非 squeeze。
"""

import math
from typing import List, Optional, Sequence

import torch

from mpo_modules.tt_ops import matrix_tt_svd

# ============================================================
# Chain-to-ring conversion helper
# ============================================================


def chain_to_ring(
    cores: List[torch.Tensor],
    ring_rank: int,
    eps_noise: float = 0.0,
) -> List[torch.Tensor]:
    """
    将 open-boundary chain 核转换为 ring (periodic) 核。

    Open cores 形状:
        [1, o_0, i_0, r_0], [r_0, o_1, i_1, r_1], ..., [r_{K-2}, o_{K-1}, i_{K-1}, 1]

    Ring cores 形状:
        [ring_rank, o_0, i_0, r_0], ..., [r_{K-2}, o_{K-1}, i_{K-1}, ring_rank]

    策略: 首核 α=0 切片保留原信息，α>0 默认零填充（确定性）。
    若 eps_noise > 0，则 α>0 用 RMS 缩放的小噪声填充（训练时可用）。
    收缩后 Tr_{ring}(G'_1 ... G'_K) = W（零填充时精确）。

    Args:
        cores: open-boundary chain 核列表
        ring_rank: 周期 bond 大小
        eps_noise: 噪声尺度（相对于核 RMS），默认 0（零填充，确定性）

    Returns:
        ring 核列表
    """
    if len(cores) < 2:
        return cores

    K = len(cores)
    ring_rank = max(1, int(ring_rank))
    device = cores[0].device
    dtype = cores[0].dtype

    # --- 处理首核 ---
    first = cores[0]  # [1, o₁, i₁, r₁]
    _, o1, i1, r1 = first.shape
    new_first = torch.zeros(ring_rank, o1, i1, r1, device=device, dtype=dtype)
    new_first[0] = first[0]
    if ring_rank > 1 and eps_noise > 0:
        rms = float(first.float().pow(2).mean().sqrt()) + 1e-12
        noise = torch.randn(ring_rank - 1, o1, i1, r1, device=device, dtype=dtype)
        new_first[1:] = noise * (eps_noise * rms)

    # --- 处理尾核 ---
    last = cores[-1]  # [r_{K-1}, o_K, i_K, 1]
    rK, oK, iK, _ = last.shape
    new_last = torch.zeros(rK, oK, iK, ring_rank, device=device, dtype=dtype)
    new_last[:, :, :, 0] = last[:, :, :, 0]
    if ring_rank > 1 and eps_noise > 0:
        rms = float(last.float().pow(2).mean().sqrt()) + 1e-12
        noise = torch.randn(rK, oK, iK, ring_rank - 1, device=device, dtype=dtype)
        new_last[:, :, :, 1:] = noise * (eps_noise * rms)

    # 组装 ring 核列表
    ring_cores: List[torch.Tensor] = [new_first]
    for k in range(1, K - 1):
        ring_cores.append(cores[k])
    ring_cores.append(new_last)

    return ring_cores


# ============================================================
# Ring SVD 分解 (chain-then-close)
# ============================================================


@torch.no_grad()
def matrix_ring_svd(
    A: torch.Tensor,
    in_factors: Sequence[int],
    max_rank: int = 64,
    svd_tol: Optional[float] = 1e-4,
    ring_rank: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    将方阵 A (n×n) 分解为 ring MPO 核（周期边界条件）。

    算法 "chain-then-close":
    1. 调用 matrix_tt_svd 得到 open-boundary chain 核
    2. 将首尾边界维（均为 1）扩展为 ring_rank 大小的周期性虚 bond
    3. 首核 α=0 切片保留原信息，α>0 用小噪声填充（训练时自适应）
    4. 满足 Tr_{α}(G'₁ · G₂ · ... · G'_K) ≈ W + O(ε)

    Args:
        A: 方阵 [n, n]
        in_factors: 每个模式的大小，满足 ∏ in_factors = n
        max_rank: chain TT 阶段的最大秩
        svd_tol: SVD 截断容差（能量阈值）
        ring_rank: 周期性虚 bond 的大小，默认为 min(4, max_rank)

    Returns:
        Ring MPO 核列表，每核形状 [r_{k-1}, o_k, i_k, r_k]
        其中 cores[0].shape[0] == cores[-1].shape[-1] == ring_rank

    Example:
        >>> A = torch.randn(64, 64)
        >>> cores = matrix_ring_svd(A, in_factors=[8, 8], max_rank=16, ring_rank=4)
        >>> print([c.shape for c in cores])
        [torch.Size([4, 8, 8, 16]), torch.Size([16, 8, 8, 4])]
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A 必须为方阵"
    n = int(A.shape[0])
    prod = 1
    for s in in_factors:
        prod *= int(s)
    assert prod == n, f"∏ in_factors={prod} 与 A.size={n} 不一致"

    K = len(in_factors)
    assert K >= 2, "ring 分解至少需要 2 个模式"

    device = A.device
    dtype = A.dtype

    if ring_rank is None:
        ring_rank = min(4, max_rank)
    ring_rank = max(1, int(ring_rank))

    # Step 1: chain TT 分解（open boundary，首尾 bond 为 1）
    chain_cores = matrix_tt_svd(A, in_factors, max_rank=max_rank, svd_tol=svd_tol)

    # Step 2: 将首尾边界维扩展为 ring_rank
    ring_cores = chain_to_ring(chain_cores, ring_rank=ring_rank)

    return ring_cores


# ============================================================
# Ring 核收缩（还原 dense 权重）
# ============================================================


def contract_ring(cores: List[torch.Tensor], order: str = "oi") -> torch.Tensor:
    """
    安全且高效的张量环 (Tensor Ring) 收缩算法。
    通过优化收缩路径，在乘入最后一个核的同时消除首尾键维，彻底规避 OOM。
    """
    assert len(cores) >= 2, "ring 收缩至少需要 2 个核"
    assert order in ("oi", "io"), f"order 必须为 'oi' 或 'io'，收到 '{order}'"

    ring_bond_left = int(cores[0].shape[0])
    ring_bond_right = int(cores[-1].shape[-1])
    assert ring_bond_left == ring_bond_right, (
        f"周期 bond 不一致：首核左={ring_bond_left}, 尾核右={ring_bond_right}"
    )

    K = len(cores)
    
    # =========================================================
    # 核心安全收缩：人为硬编码最优路径 (K=2, 3 最常见)
    # =========================================================
    if K == 2:
        # cores[0]: [a, o1, i1, b], cores[1]: [b, o2, i2, a]
        x = torch.einsum('aoib, bpja -> oipj', cores[0], cores[1])
        
    elif K == 3:
        # cores[0]: [a, o1, i1, b], cores[1]: [b, o2, i2, c], cores[2]: [c, o3, i3, a]
        tmp = torch.einsum('aoib, bpjc -> aoipjc', cores[0], cores[1])
        x = torch.einsum('aoipjc, cqka -> oipjqk', tmp, cores[2])
        
    else:
        # =========================================================
        # 动态回退方案 (K >= 4): 保证内存安全的同时支持任意长度
        # =========================================================
        tmp = cores[0]  # [a, o1, i1, b]
        
        # 1. 连乘前 K-1 个核 (把除最后一个核之外的所有张量乘起来)
        # 此时 tmp 的形状最大仅为 [a, o1, i1, ..., o_{K-1}, i_{K-1}, x]
        # 因为缺少了最后一维，整体体积缩小了 (o_K * i_K) 倍，完全在安全范围内！
        for c in cores[1:-1]:
            tmp = torch.tensordot(tmp, c, dims=([-1], [0]))
            
        c_last = cores[-1]
        
        # 2. 动态生成 einsum 下标，在最后一步同时 trace 掉头尾的 'a'
        ndim = tmp.ndim
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        idx = 0
        
        head_char = chars[idx]; idx += 1                        # a
        mid_chars = chars[idx : idx + ndim - 2]; idx += ndim -2 # o1, i1, ..., o_{K-1}, i_{K-1}
        tail_char = chars[idx]; idx += 1                        # x (中间的最后一个 bond)
        
        last_mid_chars = chars[idx : idx + 2]; idx += 2         # o_K, i_K
        
        lhs1 = head_char + mid_chars + tail_char                # 例: a b c d e f
        lhs2 = tail_char + last_mid_chars + head_char           # 例: f g h a
        rhs  = mid_chars + last_mid_chars                       # 例: b c d e g h (完美对齐 o, i 交替)
        
        x = torch.einsum(f"{lhs1},{lhs2}->{rhs}", tmp, c_last)

    # =========================================================
    # 格式对齐：恢复为 Dense 的 [out_f, in_f]
    # =========================================================
    num = len(cores)
    if order == "oi":
        # 偶数位(0,2,...)=o 轴，奇数位(1,3,...)=i 轴
        perm = list(range(0, 2 * num, 2)) + list(range(1, 2 * num, 2))
        x = x.permute(*perm).contiguous()
        o_shape = x.shape[:num]
        i_shape = x.shape[num:]
        W = x.reshape(
            int(math.prod(o_shape)),
            int(math.prod(i_shape)),
        )
    else:
        perm = list(range(1, 2 * num, 2)) + list(range(0, 2 * num, 2))
        x = x.permute(*perm).contiguous()
        i_shape = x.shape[:num]
        o_shape = x.shape[num:]
        W = x.reshape(
            int(math.prod(i_shape)),
            int(math.prod(o_shape)),
        ).t()

    return W

# ============================================================
# Ring 秩信息查询
# ============================================================


def get_ring_rank_info(cores: List[torch.Tensor]) -> dict:
    """
    获取 ring MPO 核的秩信息。

    Args:
        cores: ring MPO 核列表

    Returns:
        包含秩信息的字典:
        - num_cores: 核数量
        - ranks: 内部 bond 秩列表（长度 K-1）
        - max_rank: 最大内部 bond 秩
        - ring_bond: 周期 bond 大小
        - shapes: 每核形状元组列表
    """
    if not cores:
        return {"num_cores": 0, "ranks": [], "max_rank": 0, "ring_bond": 0}

    ranks = []
    for i in range(len(cores) - 1):
        ranks.append(int(cores[i].shape[-1]))

    ring_bond = int(cores[0].shape[0])

    return {
        "num_cores": len(cores),
        "ranks": ranks,
        "max_rank": max(ranks) if ranks else 0,
        "ring_bond": ring_bond,
        "shapes": [tuple(c.shape) for c in cores],
    }
