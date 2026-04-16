#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Modules - MPO 分解算法

包含将线性层分解为 MPO 的核心算法。
从 mpo_utils.py (行 1621-3400) 迁移而来。

这是最复杂的模块之一，包含多种分解策略和白化方法。
"""

import os
from typing import List, Tuple

import torch
import torch.nn as nn


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


def _select_svd_rank(
    svals: torch.Tensor,
    *,
    chi_cap: int,
    rank_avail: int,
    rel_amp_tol: float,
    energy_rel_tol: float,
    energy_abs_tol: float,
) -> tuple[int, dict]:
    """
    统一的截断规则：
      1) rank_avail：受矩阵维度限制的上限
      2) rel_amp_tol：按奇异值幅度（相对最大值）筛掉极小成分
      3) energy_*：按能量阈值（累计 σ^2）或绝对阈值控制
      4) chi_cap：最终 χ 上限
    返回 (selected_rank, debug_info)
    """
    info = {
        "rank_avail": int(rank_avail),
        "rel_amp_tol": float(rel_amp_tol),
        "energy_rel_tol": float(energy_rel_tol),
        "energy_abs_tol": float(energy_abs_tol),
        "rel_rank": None,
        "energy_rank": None,
    }
    if svals.numel() == 0:
        return 1, info

    rank_cap = max(1, int(rank_avail))

    rel_rank = rank_cap
    if rel_amp_tol > 0:
        smax = float(svals[0])
        if smax > 0:
            keep = int((svals > (rel_amp_tol * smax)).sum().item())
            rel_rank = max(1, keep) if keep > 0 else 1
    info["rel_rank"] = int(rel_rank)

    energy_rank = rank_cap
    if (energy_rel_tol > 0 or energy_abs_tol > 0) and svals.numel() > 0:
        sq = svals.to(torch.float64) ** 2
        total = float(sq.sum().item())
        if energy_rel_tol > 0 and total > 0:
            thr = (1.0 - min(1.0, max(0.0, energy_rel_tol))) * total
            if thr <= 0:
                energy_rank = 1
            elif thr >= total:
                energy_rank = min(rank_cap, svals.numel())
            else:
                cumsum = torch.cumsum(sq, dim=0)
                thr_tensor = cumsum.new_tensor(thr)
                idx = int(torch.searchsorted(cumsum, thr_tensor, right=False).item())
                energy_rank = max(1, min(idx + 1, svals.numel()))
        if energy_abs_tol > 0:
            abs_rank = int((svals > energy_abs_tol).sum().item())
            if abs_rank > 0:
                energy_rank = min(energy_rank, abs_rank)
            else:
                energy_rank = 1
    info["energy_rank"] = int(energy_rank)

    r = max(1, min(rank_cap, int(chi_cap), int(rel_rank), int(energy_rank)))
    info["selected_rank"] = int(r)
    return r, info


def _sanitize_cores(
    cores: List[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    abs_warn: float,
    abs_clamp: float,
) -> List[torch.Tensor]:
    cleaned = []
    summary_vals = []
    any_warn = False
    for idx, core in enumerate(cores):
        if not torch.isfinite(core).all():
            core = torch.nan_to_num(core, nan=0.0, posinf=1e4, neginf=-1e4)
        max_abs = float(core.abs().max().item()) if core.numel() > 0 else 0.0
        summary_vals.append(max_abs)
        if abs_warn > 0 and max_abs > abs_warn:
            print(f"[warn] MPO 核 {idx} abs≈{max_abs:.2e} 超出警戒 {abs_warn:g}")
            any_warn = True
        if abs_clamp > 0 and max_abs > abs_clamp:
            core = torch.clamp(core, min=-abs_clamp, max=abs_clamp)
            print(f"[info] MPO 核 {idx} abs 已裁剪至 ±{abs_clamp:g}")
        cleaned.append(core.to(device=device, dtype=dtype))
    # 若任意核触发告警，附带打印所有核心的 max_abs，便于对比
    if any_warn:
        try:
            detail = " ".join([f"{i}:{v:.2e}" for i, v in enumerate(summary_vals)])
            print(f"[info] MPO 核各 max_abs: {detail}")
        except Exception:
            pass
    return cleaned


def _gauge_balance_tt_cores(
    cores: List[torch.Tensor],
    *,
    max_scale: float = 8.0,
    eps: float = 1e-8,
    iters: int = 1,
) -> List[torch.Tensor]:
    """
    利用 TT 的 gauge 自由度，在相邻核之间沿内部秩 r_k 进行尺度再分配，
    以降低单个核心的峰值幅度而不改变整体算子（数值等价）。

    做法：对每个键 r_k，把左核在该键上的列范数规范到统一尺度，
    并将相反的缩放乘回到右核的行上，从而保持等价。

    参数：
      - max_scale: 单次在一个键上允许的最大缩放因子（>1），用于避免过度拉伸/压缩
      - eps: 防止除零的下限
      - iters: 迭代次数（>1 时可进一步平衡）
    """
    if not isinstance(cores, list) or len(cores) <= 1:
        return cores

    K = len(cores)

    # 仅在数值合法时执行，避免引入 NaN/Inf
    def _is_finite_all(ts: torch.Tensor) -> bool:
        try:
            return bool(torch.isfinite(ts).all())
        except Exception:
            return True

    max_scale = float(max(1.0, max_scale))
    eps = float(max(eps, 1e-12))
    iters = max(1, int(iters))

    for _ in range(iters):
        for k in range(K - 1):
            Gk = cores[k]
            Gn = cores[k + 1]
            if not (_is_finite_all(Gk) and _is_finite_all(Gn)):
                continue
            if Gk.ndim != 4 or Gn.ndim != 4:
                continue
            r_prev, ok, ik, r_next = map(int, Gk.shape)
            r_next2, ok2, ik2, r2 = map(int, Gn.shape)
            if r_next != r_next2:
                # 形状异常（不匹配），跳过该键
                continue

            # 计算 Gk 在 r_next 维的列范数，并拉回到统一尺度
            A = Gk.reshape(r_prev * ok * ik, r_next).to(torch.float32)
            col_norm = torch.linalg.norm(A, ord=2, dim=0)  # [r_next]
            # 目标尺度：均值范数（稳健，可改为中位数）
            target = torch.clamp(col_norm.mean(), min=eps)
            scale = torch.clamp(col_norm / target, min=1.0 / max_scale, max=max_scale)

            # 左核：在 r_next 维按 1/scale 缩放列
            inv_scale = (1.0 / scale).to(A.dtype)
            A = A * inv_scale.view(1, -1)
            Gk_new = A.reshape(r_prev, ok, ik, r_next)

            # 右核：在 r_prev（=该键）维按 scale 放大对应行，保持等价
            B = Gn.reshape(r_next, ok2 * ik2 * r2).to(torch.float32)
            B = scale.view(-1, 1) * B
            Gn_new = B.reshape(r_next, ok2, ik2, r2)

            cores[k] = Gk_new.to(dtype=Gk.dtype, device=Gk.device)
            cores[k + 1] = Gn_new.to(dtype=Gn.dtype, device=Gn.device)

    return cores


def _gauge_norm_three_cores(
    cores: List[torch.Tensor],
    target: float,
    max_iters: int = 6,
    *,
    step_mode: str = "sqrt",  # 'sqrt' 或 'direct'
    neighbor_policy: str = "min",  # 'min' 或 'both'
) -> List[torch.Tensor]:
    """
    针对 K=3 的 MPO，利用标量 gauge 在三个核之间重新分配尺度，
    将所有核的 max_abs 约束在 target 附近。target<=0 时无操作。
    """
    if not isinstance(cores, list) or len(cores) != 3:
        return cores
    # 占位实现：若需要更强的三核规范化，可在此扩展。
    # 目前保持传入 cores 不变，由后续的 Stiefel 规范化负责主要均衡工作。
    return cores


def _gauge_stiefel_three_cores(
    cores: List[torch.Tensor],
    *,
    smin: float = 0.2,
    smax: float = 5.0,
    iters: int = 1,
) -> List[torch.Tensor]:
    """
    针对 K=3 的 MPO，在吸收完成后进行一次“近幺正”规范化：
      1) 左核在 bond 维做列正交（G1: QR）并将 R 乘回到 G2 左侧
      2) 右核在 bond 维做行正交（G3^T: QR）并将 R^T 乘回到 G2 右侧
      3) 中核在 bond 维做一次 SVD，并将奇异值裁剪到 [smin, smax] 之后重建

    注意：这是数值规范化步骤，可能对算子产生极小扰动（取决于裁剪门限）。
    """
    if not isinstance(cores, list) or len(cores) != 3:
        return cores
    G1, G2, G3 = cores
    if not (torch.is_tensor(G1) and torch.is_tensor(G2) and torch.is_tensor(G3)):
        return cores

    # 快速形状校验
    try:
        chi = int(G2.shape[0])
        assert int(G1.shape[-1]) == chi and int(G3.shape[0]) == chi and int(G3.shape[-1]) == 1
    except Exception:
        return cores

    smin = float(max(1e-8, smin))
    smax = float(max(smin, smax))
    iters = max(1, int(iters))

    device = G2.device
    dtype = G2.dtype

    def _qr_stable(A: torch.Tensor):
        try:
            return torch.linalg.qr(A, mode="reduced")
        except Exception:
            A64 = A.detach().to(device="cpu", dtype=torch.float64)
            Q, R = torch.linalg.qr(A64, mode="reduced")
            return Q.to(device=A.device, dtype=A.dtype), R.to(device=A.device, dtype=A.dtype)

    def _svd_stable(A: torch.Tensor):
        try:
            # 优先使用gesvd driver避免cuSOLVER问题
            return torch.linalg.svd(A, full_matrices=False, driver="gesvd")
        except (RuntimeError, TypeError):
            try:
                return torch.linalg.svd(A, full_matrices=False)
            except Exception:
                A64 = A.detach().to(device="cpu", dtype=torch.float64)
                U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
                return (
                    U.to(device=A.device, dtype=A.dtype),
                    S.to(device=A.device, dtype=A.dtype),
                    Vh.to(device=A.device, dtype=A.dtype),
                )

    for _ in range(iters):
        # Step 1: 左核列正交（G1: QR）
        o1, i1 = int(G1.shape[1]), int(G1.shape[2])
        M1 = G1.reshape(o1 * i1, chi).to(dtype=torch.float32)
        M1 = torch.nan_to_num(M1, nan=0.0, posinf=1e6, neginf=-1e6)
        Q1, R1 = _qr_stable(M1)
        Q1 = Q1[:, :chi]
        R1 = R1[:chi, :chi]
        G1 = Q1.reshape(1, o1, i1, chi).to(device=device, dtype=dtype)
        # 将 R1 乘回 G2 左侧
        M2 = G2.reshape(chi, -1).to(dtype=torch.float32)
        M2 = torch.nan_to_num(M2, nan=0.0, posinf=1e6, neginf=-1e6)
        M2 = R1 @ M2
        G2 = M2.reshape(chi, int(G2.shape[1]), int(G2.shape[2]), int(G2.shape[3])).to(device=device, dtype=dtype)

        # Step 2: 右核行正交（对 G3^T 做 QR）
        o3, i3 = int(G3.shape[1]), int(G3.shape[2])
        M3 = G3.reshape(chi, o3 * i3).to(dtype=torch.float32)
        M3 = torch.nan_to_num(M3, nan=0.0, posinf=1e6, neginf=-1e6)
        Q3T, R3 = _qr_stable(M3.transpose(0, 1))  # [o3*i3, chi] = Q3T @ R3
        M3_new = Q3T.transpose(0, 1)
        G3 = M3_new.reshape(chi, o3, i3, 1).to(device=device, dtype=dtype)
        # 将 R3^T 乘回 G2 右侧
        M2 = G2.reshape(-1, chi).to(dtype=torch.float32)
        M2 = torch.nan_to_num(M2, nan=0.0, posinf=1e6, neginf=-1e6)
        M2 = M2 @ R3.transpose(0, 1)
        G2 = M2.reshape(chi, int(G2.shape[1]), int(G2.shape[2]), int(G2.shape[3])).to(device=device, dtype=dtype)

        # Step 3: 中核奇异值处理（可选剪裁/可跳过）
        noclip = os.getenv("MPO_GAUGE_STIEFEL_NOCLIP", "0") != "0"
        if not noclip:
            o2, i2 = int(G2.shape[1]), int(G2.shape[2])
            M2 = G2.reshape(chi, chi * o2 * i2).to(dtype=torch.float32)
            M2 = torch.nan_to_num(M2, nan=0.0, posinf=1e6, neginf=-1e6)
            U, S, Vh = _svd_stable(M2)
            # 裁剪奇异值到 [smin, smax]
            S = torch.clamp(S, min=smin, max=smax)
            M2_new = (U * S.unsqueeze(0)) @ Vh
            G2 = M2_new.reshape(chi, o2, i2, chi).to(device=device, dtype=dtype)

    return [G1, G2, G3]


# ============================================================
# SVD utilities
# ============================================================


def robust_svd_split(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced SVD split (√σ split) to reduce intermediate conditioning.

    Given A = U @ diag(S) @ Vh, return:
      - U_bal = U @ diag(sqrt(S + eps))  (implemented as elementwise scale)
      - T_next = diag(sqrt(S + eps)) @ Vh (implemented as elementwise scale)
    """
    if eps < 0:
        eps = 0.0
    S_nonneg = torch.clamp(S, min=0.0)
    S_sqrt = torch.sqrt(S_nonneg + float(eps)).to(dtype=U.dtype, device=U.device)
    U_bal = U * S_sqrt.unsqueeze(0)
    T_next = S_sqrt.unsqueeze(1) * Vh
    return U_bal, T_next


# ============================================================
# 基础 MPO 分解
# ============================================================


def factor_linear_mpo(
    linear: nn.Linear,
    bond_dim: int,
    num_cores: int,
    layer_name: str = "",
    boundary: str = "open",
    s_vector: torch.Tensor = None,
):
    """
    将线性层分解为 MPO 层（基础版本）。

    Args:
        linear: 输入的线性层
        bond_dim: MPO 键维（秩）上限
        num_cores: MPO 核数（K）
        layer_name: 层名（用于边重心轻拆法判断是否是 MLP）

    Returns:
        MPOLinear 实例

    功能：
    - 使用 SVD 进行张量分解
    - 自动清理 NaN/Inf 值
    - K=2 时退化为单次 SVD
    - 支持边重心轻拆法（通过 MPO_MLP_EDGE_HEAVY=1 启用）

    Example:
        >>> linear = nn.Linear(4096, 4096)
        >>> mpo = factor_linear_mpo(linear, bond_dim=64, num_cores=3)
        >>> print(mpo.cores[0].shape)
        torch.Size([1, out_fac[0], in_fac[0], 64])
    """
    import os
    import sys
    from pathlib import Path

    # 导入 MPOLinear
    from mpo_modules.core import MPOLinear
    from mpo_modules.factorization_utils import _find_factors_edge_heavy
    from mpo_modules.helpers import find_factors_balanced

    if num_cores < 2:
        raise ValueError("num_cores must be >= 2")

    W = linear.weight.detach().to(torch.float32)
    out_f, in_f = W.shape
    device = linear.weight.device
    dtype0 = linear.weight.dtype

    # 检查权重矩阵的有效性
    if not torch.isfinite(W).all():
        print("  ⚠️  警告：权重矩阵包含NaN/Inf，尝试清理...")
        W = torch.nan_to_num(W, nan=0.0, posinf=1e4, neginf=-1e4)

    # 确定 out_fac / in_fac（保留原有的边重心轻拆法逻辑）
    use_edge_heavy = os.getenv("MPO_MLP_EDGE_HEAVY", "0") == "1"
    if use_edge_heavy and layer_name:
        out_fac, in_fac = _find_factors_edge_heavy(out_f, in_f, num_cores, layer_name)
    else:
        out_fac = find_factors_balanced(out_f, num_cores)
        in_fac = find_factors_balanced(in_f, num_cores)

    # ---- 直接调用用户写的自定义分解函数（test_MPO.py） ----
    _project_llm = Path(__file__).resolve().parents[2]  # mpo_modules -> MPO_Compression -> LLM
    if str(_project_llm) not in sys.path:
        sys.path.insert(0, str(_project_llm))
    from test_MPO import factor_linear_mpo_custom

    cores = factor_linear_mpo_custom(W, bond_dim=bond_dim, num_cores=num_cores, out_fac=out_fac, in_fac=in_fac, s_vector=s_vector)

    # 转回原始 device/dtype
    cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores]

    # 若为周期边界，将 chain 核转换为 ring 核
    if boundary == "periodic":
        from mpo_modules.ring_ops import chain_to_ring
        cleaned_cores = chain_to_ring(cleaned_cores, ring_rank=bond_dim)

    mpo = MPOLinear(in_f, out_f, cleaned_cores, boundary=boundary)
    return mpo


def get_mpo_compression_ratio(in_features: int, out_features: int, num_cores: int, bond_dim: int) -> float:
    """
    计算 MPO 相对于 dense 层的压缩比。

    Args:
        in_features: 输入维度
        out_features: 输出维度
        num_cores: MPO 核数
        bond_dim: MPO 键维上限

    Returns:
        压缩比（MPO 参数量 / Dense 参数量）
    """
    from mpo_modules.helpers import find_factors_balanced

    dense_params = in_features * out_features

    if num_cores == 2:
        mpo_params = in_features * bond_dim + bond_dim * out_features
    else:
        out_fac = find_factors_balanced(out_features, num_cores)
        in_fac = find_factors_balanced(in_features, num_cores)

        mpo_params = 0
        mpo_params += out_fac[0] * in_fac[0] * bond_dim
        for k in range(1, num_cores - 1):
            mpo_params += bond_dim * out_fac[k] * in_fac[k] * bond_dim
        mpo_params += bond_dim * out_fac[-1] * in_fac[-1]

    return mpo_params / dense_params


def estimate_mpo_bond_dim(in_features: int, out_features: int, num_cores: int, target_ratio: float = 0.4) -> int:
    """
    根据目标压缩比估算所需的 bond_dim。
    使用精确的 MPO 参数量公式求解二次方程，避免低估。
    """
    from mpo_modules.helpers import find_factors_balanced

    dense_params = in_features * out_features
    target_params = dense_params * target_ratio

    if num_cores == 2:
        bond_dim = int(target_params / (in_features + out_features))
    else:
        out_fac = find_factors_balanced(out_features, num_cores)
        in_fac = find_factors_balanced(in_features, num_cores)
        oi = [out_fac[k] * in_fac[k] for k in range(num_cores)]
        # MPO params = b*chi + a*chi^2 where
        #   b = oi[0] + oi[-1] (linear terms at boundaries)
        #   a = sum(oi[1:-1])  (quadratic terms in the middle)
        b = oi[0] + oi[-1]
        a = sum(oi[1:-1])
        c_val = -target_params
        if a > 0:
            bond_dim = int((-b + (b * b - 4 * a * c_val) ** 0.5) / (2 * a))
        elif b > 0:
            bond_dim = int(-c_val / b)
        else:
            bond_dim = 1

    return max(1, bond_dim)


# ============================================================
# 导出
# ============================================================

__all__ = [
    "factor_linear_mpo",
    "get_mpo_compression_ratio",
    "estimate_mpo_bond_dim",
    "robust_svd_split",
]
