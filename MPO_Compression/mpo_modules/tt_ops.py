#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Modules - Tensor Train 操作

包含 TT 分解、MPO 右乘算子、TT rounding 等核心操作。
这些函数从 mpo_utils.py (行 1722-2253) 迁移而来。
"""

import os
from typing import List, Optional, Sequence

import torch

# ============================================================
# 矩阵 TT-SVD 分解
# ============================================================


@torch.no_grad()
def matrix_tt_svd(
    A: torch.Tensor, in_factors: Sequence[int], max_rank: int = 64, svd_tol: Optional[float] = 1e-4
) -> List[torch.Tensor]:
    """
    将方阵 A (n×n) 分解为 TT-operator 核，每核形状 [r_{k-1}, i_k(out), i_k(in), r_k]。

    Args:
        A: 方阵 [n, n]
        in_factors: 每个 TT 模式的大小，满足 ∏ in_factors = n
        max_rank: 最大 TT 秩
        svd_tol: SVD 截断容差（能量阈值）

    Returns:
        TT-operator 核列表，每核形状 [r_{k-1}, i_k, i_k, r_k]

    Example:
        >>> A = torch.randn(64, 64)
        >>> cores = matrix_tt_svd(A, in_factors=[8, 8], max_rank=16)
        >>> print([c.shape for c in cores])
        [torch.Size([1, 8, 8, 16]), torch.Size([16, 8, 8, 1])]
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A 必须为方阵"
    n = int(A.shape[0])
    prod = 1
    for s in in_factors:
        prod *= int(s)
    assert prod == n, f"∏ in_factors={prod} 与 A.size={n} 不一致"

    device = A.device
    dtype = A.dtype
    K = len(in_factors)

    # Reshape 并 permute 为交错格式
    T = A.reshape(*in_factors, *in_factors).permute(*[j for k in range(K) for j in (k, k + K)]).contiguous()

    cores: List[torch.Tensor] = []
    r_prev = 1

    for k in range(K - 1):
        io = int(in_factors[k])
        ii = int(in_factors[k])
        T = T.reshape(r_prev * io * ii, -1)

        # 稳健 SVD：清理 NaN/Inf，归一化后做 QR+SVD
        T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
        scale = T.abs().amax()
        T_scaled = T / scale if float(scale) > 0 else T
        Q, R = torch.linalg.qr(T_scaled, mode="reduced")

        try:
            U_r, S, Vh = torch.linalg.svd(R, full_matrices=False, driver="gesvd")
        except TypeError:
            U_r, S, Vh = torch.linalg.svd(R, full_matrices=False)
        except Exception:
            # CPU + FP64 兜底
            R64 = R.detach().to(device="cpu", dtype=torch.float64)
            U_r, S, Vh = torch.linalg.svd(R64, full_matrices=False)
            U_r = U_r.to(device=R.device, dtype=T.dtype)
            S = S.to(device=R.device, dtype=T.dtype)
            Vh = Vh.to(device=R.device, dtype=T.dtype)

        U = Q @ U_r
        S = S * scale

        # 能量阈值截断
        if S.numel():
            if svd_tol is not None and svd_tol > 0:
                sq = S * S
                total = sq.sum()
                cumsum = torch.cumsum(sq.flip(0), dim=0).flip(0)
                tail = cumsum - sq
                mask = tail <= (svd_tol**2) * total
                keep_e = int(mask.nonzero().min().item() + 1) if mask.any() else S.numel()
            else:
                keep_e = S.numel()
        else:
            keep_e = 1

        r = max(1, min(int(keep_e), int(max_rank)))
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        core = U.reshape(r_prev, io, ii, r)
        cores.append(core.to(device=device, dtype=dtype))

        T = S.unsqueeze(1) * Vh
        r_prev = r
        T = T.reshape(r_prev, *T.shape[1:])

    # 最后一核
    io = int(in_factors[-1])
    ii = int(in_factors[-1])
    cores.append(T.reshape(r_prev, io, ii, 1).to(device=device, dtype=dtype))

    return cores


def _get_intermediate_dtype() -> torch.dtype:
    if os.getenv("MPO_FULLTT_INTERMEDIATE_FP16", "0") == "1":
        return torch.float16
    if os.getenv("MPO_FULLTT_INTERMEDIATE_BF16", "0") == "1":
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float32


def contract_core_with_tt_operator(
    G: torch.Tensor,
    M: torch.Tensor,
    *,
    max_chunk_elems: Optional[int] = None,
) -> torch.Tensor:
    """contract core with TT-operator using automatic chunking."""
    if max_chunk_elems is None:
        try:
            max_chunk_elems = int(os.getenv("MPO_RIGHT_APPLY_MAX_TENSOR_ELEMS", "200000000"))
        except Exception:
            max_chunk_elems = 200_000_000
    max_chunk_elems = max(1, int(max_chunk_elems))

    a, o, i, b = G.shape
    c, io, ii, d = M.shape
    assert i == ii, f"i 维不匹配：{i} vs {ii}"
    assert io == i, f"operator io={io} 需要与 MPO i={i} 匹配"

    interm_dtype = _get_intermediate_dtype()
    total_elems = (a * c) * o * io * (b * d)
    if total_elems <= max_chunk_elems:
        T = torch.tensordot(G.to(interm_dtype), M.to(interm_dtype), dims=([2], [2]))
        return T.permute(0, 1, 3, 4, 2, 5).contiguous().reshape(a * c, o, io, b * d).to(dtype=G.dtype, device=G.device)

    # 自动调整左右秩分块，确保单块元素量受限
    chunk_left = c
    chunk_right = d

    def chunk_elems(cl: int, dr: int) -> int:
        return (a * cl) * o * io * (b * dr)

    while chunk_elems(chunk_left, chunk_right) > max_chunk_elems:
        if chunk_left >= chunk_right and chunk_left > 1:
            chunk_left = max(1, chunk_left // 2)
        elif chunk_right > 1:
            chunk_right = max(1, chunk_right // 2)
        else:
            break

    result6d = torch.zeros(a, o, c, io, b, d, dtype=interm_dtype, device=G.device)
    for c_start in range(0, c, chunk_left):
        c_end = min(c, c_start + chunk_left)
        left_chunk = M[c_start:c_end]
        for d_start in range(0, d, chunk_right):
            d_end = min(d, d_start + chunk_right)
            chunk = left_chunk[..., d_start:d_end]
            T_chunk = torch.tensordot(G.to(interm_dtype), chunk.to(interm_dtype), dims=([2], [2]))
            T_chunk = T_chunk.permute(0, 1, 3, 4, 2, 5).contiguous()
            result6d[:, :, c_start:c_end, :, :, d_start:d_end] = T_chunk

    return result6d.reshape(a * c, o, io, b * d).to(dtype=G.dtype, device=G.device)


# ============================================================
# MPO 右乘算子
# ============================================================


@torch.no_grad()
def mpo_right_apply_operator(
    cores_linear: List[torch.Tensor],
    cores_op: List[torch.Tensor],
    *,
    max_chunk_elems: Optional[int] = None,
) -> List[torch.Tensor]:
    """将 MPO 算子右乘到线性层 MPO 上，自动控制中间显存。"""
    assert len(cores_linear) == len(cores_op), "MPO 核数量不匹配"
    return [
        contract_core_with_tt_operator(G, M, max_chunk_elems=max_chunk_elems) for G, M in zip(cores_linear, cores_op)
    ]


# ============================================================
# TT Rounding（4D 核压缩）
# ============================================================


@torch.no_grad()
def tt_round_4d_cores(
    cores: List[torch.Tensor],
    chi_cap: int,
    tol: float = 1e-4,
) -> List[torch.Tensor]:
    """
    对 4D MPO 核 [r_{k-1}, o_k, i_k, r_k] 执行 TT-rounding。

    先左→右 SVD 压缩键维，再右→左 SVD 压缩键维，
    使用能量阈值 + rank cap 控制截断。

    Args:
        cores: MPO 核列表
        chi_cap: 秩上限
        tol: 能量阈值（相对于总能量）

    Returns:
        压缩后的核列表

    Note:
        - 会修改 cores（原地操作）
        - 先 L→R 再 R→L 确保正交规范
    """
    K = len(cores)
    # 允许设置每步最小秩，避免过度截断导致塌零
    try:
        _min_r = int(os.getenv("MPO_ROUND_MIN_RANK", "0") or 0)
    except Exception:
        _min_r = 0

    # ===== L→R 压缩 =====
    for k in range(K - 1):
        G = cores[k].to(torch.float32)
        a, o, i, b = G.shape
        Gm = G.reshape(a * o * i, b).contiguous()
        Gm = torch.nan_to_num(Gm, nan=0.0, posinf=0.0, neginf=0.0)

        scale = Gm.abs().amax()
        Gm_scaled = Gm / scale if float(scale) > 0 else Gm
        Q, R = torch.linalg.qr(Gm_scaled, mode="reduced")

        try:
            U_R, S, Vh = torch.linalg.svd(R, full_matrices=False, driver="gesvd")
        except TypeError:
            U_R, S, Vh = torch.linalg.svd(R, full_matrices=False)
        except Exception:
            R64 = R.detach().to(device="cpu", dtype=torch.float64)
            U_R, S, Vh = torch.linalg.svd(R64, full_matrices=False)
            U_R = U_R.to(device=Gm.device, dtype=torch.float32)
            S = S.to(device=Gm.device, dtype=torch.float32)
            Vh = Vh.to(device=Gm.device, dtype=torch.float32)

        U = Q @ U_R
        S = S * scale

        if S.numel() and tol > 0:
            sq = S * S
            total = sq.sum()
            cumsum = torch.cumsum(sq.flip(0), dim=0).flip(0)
            tail = cumsum - sq
            mask = tail <= (tol**2) * total
            keep_e = int(mask.nonzero().min().item() + 1) if mask.any() else S.numel()
        else:
            keep_e = S.numel() if S.numel() else 1

        r = max(1, min(int(keep_e), int(chi_cap)))
        if _min_r > 0:
            r = max(r, min(_min_r, int(chi_cap)))
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        cores[k] = U.reshape(a, o, i, r).to(dtype=cores[k].dtype, device=cores[k].device)
        SV = S.unsqueeze(1) * Vh

        Gn = cores[k + 1].to(torch.float32)
        b2, o2, i2, c2 = Gn.shape
        assert b2 == SV.shape[1], f"round L2R 键维不对齐：{b2} vs {SV.shape[1]}"
        Gn = torch.tensordot(SV, Gn, dims=([1], [0]))
        cores[k + 1] = Gn.to(dtype=cores[k + 1].dtype, device=cores[k + 1].device)

    # ===== R→L 压缩 =====
    for k in range(K - 1, 0, -1):
        G = cores[k].to(torch.float32)
        a, o, i, b = G.shape
        Gm = G.reshape(a, o * i * b).T.contiguous()
        Gm = torch.nan_to_num(Gm, nan=0.0, posinf=0.0, neginf=0.0)

        scale = Gm.abs().amax()
        Gm_scaled = Gm / scale if float(scale) > 0 else Gm
        Q, R = torch.linalg.qr(Gm_scaled, mode="reduced")

        try:
            U_R, S, Vh = torch.linalg.svd(R, full_matrices=False, driver="gesvd")
        except TypeError:
            U_R, S, Vh = torch.linalg.svd(R, full_matrices=False)
        except Exception:
            R64 = R.detach().to(device="cpu", dtype=torch.float64)
            U_R, S, Vh = torch.linalg.svd(R64, full_matrices=False)
            U_R = U_R.to(device=Gm.device, dtype=torch.float32)
            S = S.to(device=Gm.device, dtype=torch.float32)
            Vh = Vh.to(device=Gm.device, dtype=torch.float32)

        U = Q @ U_R
        S = S * scale

        if S.numel() and tol > 0:
            sq = S * S
            total = sq.sum()
            cumsum = torch.cumsum(sq.flip(0), dim=0).flip(0)
            tail = cumsum - sq
            mask = tail <= (tol**2) * total
            keep_e = int(mask.nonzero().min().item() + 1) if mask.any() else S.numel()
        else:
            keep_e = S.numel() if S.numel() else 1

        r = max(1, min(int(keep_e), int(chi_cap)))
        if _min_r > 0:
            r = max(r, min(_min_r, int(chi_cap)))
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        US = U * S.unsqueeze(0)
        G_new = US.T.reshape(r, o, i, b)
        cores[k] = G_new.to(dtype=cores[k].dtype, device=cores[k].device)

        VT = Vh.T
        Gp = cores[k - 1].to(torch.float32)
        ap, o_, i_, a2 = Gp.shape
        assert a2 == VT.shape[0], f"round R2L 键维不对齐：{a2} vs {VT.shape[0]}"
        Gp = torch.tensordot(Gp, VT, dims=([3], [0]))
        cores[k - 1] = Gp.to(dtype=cores[k - 1].dtype, device=cores[k - 1].device)

    return cores


def _round_right_matrix(T: torch.Tensor, chi_limit: int, tol_val: float) -> tuple[torch.Tensor, torch.Tensor]:
    L, o, io, R = [int(x) for x in T.shape]
    Gm = T.reshape(L * o * io, R).contiguous()
    Gm = torch.nan_to_num(Gm, nan=0.0, posinf=0.0, neginf=0.0)
    Gm32 = Gm.to(torch.float32)
    scale = Gm32.abs().amax()
    Gm_scaled = Gm32 / scale if float(scale) > 0 else Gm32
    Q, Rm = torch.linalg.qr(Gm_scaled, mode="reduced")
    try:
        U_R, S, Vh = torch.linalg.svd(Rm, full_matrices=False, driver="gesvd")
    except TypeError:
        U_R, S, Vh = torch.linalg.svd(Rm, full_matrices=False)
    except Exception:
        R64 = Rm.detach().to(device="cpu", dtype=torch.float64)
        U_R, S, Vh = torch.linalg.svd(R64, full_matrices=False)
        U_R = U_R.to(device=Gm32.device, dtype=torch.float32)
        S = S.to(device=Gm32.device, dtype=torch.float32)
        Vh = Vh.to(device=Gm32.device, dtype=torch.float32)
    U = Q @ U_R
    S = S * scale
    if S.numel() and tol_val > 0:
        sq = S * S
        total = sq.sum()
        cumsum = torch.cumsum(sq.flip(0), dim=0).flip(0)
        tail = cumsum - sq
        mask = tail <= (tol_val**2) * total
        keep_e = int(mask.nonzero().min().item() + 1) if mask.any() else int(S.numel())
    else:
        keep_e = int(S.numel()) if S.numel() else 1
    try:
        _min_r = int(os.getenv("MPO_ROUND_MIN_RANK", "0") or 0)
    except Exception:
        _min_r = 0
    r = max(1, min(int(keep_e), int(chi_limit)))
    if _min_r > 0:
        r = max(r, min(_min_r, int(chi_limit)))
    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]
    core32 = U.reshape(L, o, io, r)
    carry32 = S.unsqueeze(1) * Vh
    return core32, carry32


def _round_right_ctg(
    G: torch.Tensor,
    M: torch.Tensor,
    left_proj: Optional[torch.Tensor],
    chi_limit: int,
    tol_val: float,
    oversample: Optional[int] = None,
    power_iter: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import cotengra as ctg
        import opt_einsum as oe
    except Exception as e:
        raise RuntimeError("cotengra/opt_einsum required for streaming cotengra path") from e

    a_sz, o_sz, i_sz, b_sz = map(int, G.shape)
    c_sz, io_sz, ii_sz, d_sz = map(int, M.shape)
    assert i_sz == ii_sz, "i 维不匹配"

    acc_dt = _get_intermediate_dtype()
    if oversample is None:
        oversample = int(os.getenv("MPO_FULLTT_RAND_OVERSAMPLE", "8"))
    if power_iter is None:
        power_iter = int(os.getenv("MPO_FULLTT_RAND_POWER", "0"))
    s_target = int(min(max(1, chi_limit + int(oversample)), b_sz * d_sz))

    max_time = float(os.getenv("MPO_CTG_MAXTIME", "0.2"))
    max_repeats = int(os.getenv("MPO_CTG_REPEATS", "16"))
    minimize = os.getenv("MPO_CTG_TARGET", "size").strip().lower()
    parallel_opt = os.getenv("MPO_CTG_PARALLEL", "auto").strip().lower()
    if parallel_opt in ("true", "1", "auto"):
        parallel = True
    elif parallel_opt in ("false", "0"):
        parallel = False
    else:
        try:
            parallel = int(parallel_opt)
        except Exception:
            parallel = True
    try:
        optimizer = ctg.HyperOptimizer(
            max_time=max_time,
            max_repeats=max_repeats,
            minimize=minimize if minimize in ("flops", "size") else "size",
            parallel=parallel,
            progbar=False,
        )
    except Exception:
        optimizer = "greedy"

    Omega = torch.randn(b_sz * d_sz, s_target, device=G.device, dtype=acc_dt).view(b_sz, d_sz, s_target)
    use_left = left_proj is not None
    if use_left:
        r_prev = int(left_proj.shape[0])
        P = left_proj.to(device=G.device, dtype=acc_dt).view(r_prev, a_sz, c_sz)
        eq1 = "a o i b, c u i d, b d s, r a c -> r o u s"
        expr1 = oe.contract_expression(
            eq1,
            (a_sz, o_sz, i_sz, b_sz),
            (c_sz, io_sz, i_sz, d_sz),
            (b_sz, d_sz, s_target),
            (r_prev, a_sz, c_sz),
            optimize=optimizer,
        )
        Y = expr1(G.to(acc_dt), M.to(acc_dt), Omega, P, backend="torch")
        Ym = Y.reshape(r_prev * o_sz * io_sz, s_target).to(torch.float32)
    else:
        eq1 = "a o i b, c u i d, b d s -> a o c u s"
        expr1 = oe.contract_expression(
            eq1, (a_sz, o_sz, i_sz, b_sz), (c_sz, io_sz, i_sz, d_sz), (b_sz, d_sz, s_target), optimize=optimizer
        )
        Y = expr1(G.to(acc_dt), M.to(acc_dt), Omega, backend="torch")
        Ym = Y.reshape(a_sz * o_sz * c_sz * io_sz, s_target).to(torch.float32)

    for _ in range(max(0, int(power_iter))):
        if use_left:
            eq2 = "a o i b, c u i d, r o u s, r a c -> b d s"
            expr2 = oe.contract_expression(
                eq2,
                (a_sz, o_sz, i_sz, b_sz),
                (c_sz, io_sz, i_sz, d_sz),
                (r_prev, o_sz, io_sz, s_target),
                (r_prev, a_sz, c_sz),
                optimize=optimizer,
            )
            Z = expr2(G.to(acc_dt), M.to(acc_dt), Y, P, backend="torch")
        else:
            eq2 = "a o i b, c u i d, a o c u s -> b d s"
            expr2 = oe.contract_expression(
                eq2,
                (a_sz, o_sz, i_sz, b_sz),
                (c_sz, io_sz, i_sz, d_sz),
                (a_sz, o_sz, c_sz, io_sz, s_target),
                optimize=optimizer,
            )
            Z = expr2(G.to(acc_dt), M.to(acc_dt), Y, backend="torch")
        if use_left:
            eq3 = "a o i b, c u i d, b d s, r a c -> r o u s"
            expr3 = oe.contract_expression(
                eq3,
                (a_sz, o_sz, i_sz, b_sz),
                (c_sz, io_sz, i_sz, d_sz),
                (b_sz, d_sz, s_target),
                (r_prev, a_sz, c_sz),
                optimize=optimizer,
            )
            Y = expr3(G.to(acc_dt), M.to(acc_dt), Z, P, backend="torch")
            Ym = Y.reshape(r_prev * o_sz * io_sz, s_target).to(torch.float32)
        else:
            eq3 = "a o i b, c u i d, b d s -> a o c u s"
            expr3 = oe.contract_expression(
                eq3, (a_sz, o_sz, i_sz, b_sz), (c_sz, io_sz, i_sz, d_sz), (b_sz, d_sz, s_target), optimize=optimizer
            )
            Y = expr3(G.to(acc_dt), M.to(acc_dt), Z, backend="torch")
            Ym = Y.reshape(a_sz * o_sz * c_sz * io_sz, s_target).to(torch.float32)
        Ym = torch.linalg.qr(Ym, mode="reduced").Q

    Qm = torch.linalg.qr(Ym, mode="reduced").Q
    if use_left:
        Qten = Qm.view(r_prev, o_sz, io_sz, -1).to(dtype=acc_dt)
        eqB = "a o i b, c u i d, r o u t, r a c -> b d t"
        exprB = oe.contract_expression(
            eqB,
            (a_sz, o_sz, i_sz, b_sz),
            (c_sz, io_sz, i_sz, d_sz),
            (r_prev, o_sz, io_sz, Qten.shape[-1]),
            (r_prev, a_sz, c_sz),
            optimize=optimizer,
        )
        BT = exprB(G.to(acc_dt), M.to(acc_dt), Qten, P, backend="torch")
    else:
        Qten = Qm.view(a_sz, o_sz, c_sz, io_sz, -1).to(dtype=acc_dt)
        eqB = "a o i b, c u i d, a o c u t -> b d t"
        exprB = oe.contract_expression(
            eqB,
            (a_sz, o_sz, i_sz, b_sz),
            (c_sz, io_sz, i_sz, d_sz),
            (a_sz, o_sz, c_sz, io_sz, Qten.shape[-1]),
            optimize=optimizer,
        )
        BT = exprB(G.to(acc_dt), M.to(acc_dt), Qten, backend="torch")

    B = BT.permute(2, 0, 1).reshape(Qten.shape[-1], b_sz * d_sz)
    B_fp32 = B.to(torch.float32)
    try:
        U_t32, S32, Vh32 = torch.linalg.svd(B_fp32, full_matrices=False, driver="gesvd")
    except TypeError:
        U_t32, S32, Vh32 = torch.linalg.svd(B_fp32, full_matrices=False)
    if S32.numel() and tol_val > 0:
        sq = S32 * S32
        total = sq.sum()
        cumsum = torch.cumsum(sq.flip(0), dim=0).flip(0)
        tail = cumsum - sq
        mask = tail <= (tol_val**2) * total
        keep_e = int(mask.nonzero().min().item() + 1) if mask.any() else int(S32.numel())
    else:
        keep_e = int(S32.numel()) if S32.numel() else 1
    r = max(1, min(int(keep_e), int(chi_limit)))
    U_t32 = U_t32[:, :r]
    S32 = S32[:r]
    Vh32 = Vh32[:r, :]
    if use_left:
        core32 = (Qm @ U_t32).view(r_prev, o_sz, io_sz, r).to(acc_dt)
    else:
        core32 = (Qm @ U_t32).view(a_sz * c_sz, o_sz, io_sz, r).to(acc_dt)
    carry32 = (S32.unsqueeze(1) * Vh32).to(acc_dt)
    return core32.to(dtype=G.dtype), carry32.to(dtype=G.dtype)


@torch.no_grad()
def mpo_right_apply_operator_streaming(
    cores_linear: List[torch.Tensor],
    cores_op: List[torch.Tensor],
    chi_cap: int,
    tol: float = 1e-4,
    *,
    prefer_cotengra: Optional[bool] = None,
) -> List[torch.Tensor]:
    """流式将 TT-operator 右乘到 MPO，避免一次性物化巨大张量。"""
    assert len(cores_linear) == len(cores_op), "MPO 核数量不匹配"
    if prefer_cotengra is None:
        prefer_cotengra = os.getenv("MPO_FULLTT_CTG_ROUND", "0") == "1"

    new_cores: List[torch.Tensor] = []
    left_proj: Optional[torch.Tensor] = None
    for k, (G, M) in enumerate(zip(cores_linear, cores_op)):
        is_last = k == len(cores_linear) - 1
        if prefer_cotengra:
            try:
                if not is_last:
                    core_k, left_proj = _round_right_ctg(G, M, left_proj, int(chi_cap), float(tol))
                    new_cores.append(core_k.to(dtype=G.dtype, device=G.device))
                    continue
                else:
                    T = contract_core_with_tt_operator(G, M)
                    if left_proj is not None:
                        T = torch.tensordot(left_proj.to(T.dtype), T, dims=([1], [0]))
                    last_core = T.reshape(T.shape[0], T.shape[1], T.shape[2], 1)
                    new_cores.append(last_core.to(dtype=G.dtype, device=G.device))
                    continue
            except Exception as e:
                print(f"[stream] cotengra 路径失败，回退到经典路径: {e}")
                prefer_cotengra = False

        T = contract_core_with_tt_operator(G, M)
        if left_proj is not None:
            T = torch.tensordot(left_proj.to(T.dtype), T, dims=([1], [0]))
        if not is_last:
            core_k, left_proj = _round_right_matrix(T, chi_cap, tol)
            new_cores.append(core_k.to(dtype=G.dtype, device=G.device))
        else:
            last_core = T.reshape(T.shape[0], T.shape[1], T.shape[2], 1)
            new_cores.append(last_core.to(dtype=G.dtype, device=G.device))

    return new_cores


# ============================================================
# 工具函数
# ============================================================


def get_tt_rank_info(cores: List[torch.Tensor]) -> dict:
    """
    获取 TT 核的秩信息

    Args:
        cores: TT 核列表

    Returns:
        包含秩信息的字典
    """
    if not cores:
        return {"num_cores": 0, "ranks": [], "max_rank": 0}

    ranks = []
    for i in range(len(cores) - 1):
        r = cores[i].shape[-1]  # 右秩
        ranks.append(int(r))

    return {
        "num_cores": len(cores),
        "ranks": ranks,
        "max_rank": max(ranks) if ranks else 0,
        "shapes": [tuple(c.shape) for c in cores],
    }


# ============================================================
# 未来扩展
# ============================================================

# 可以添加更多 TT 操作：
# - TT 加法
# - TT Hadamard 积
# - TT 矩阵向量乘
# - TT 近似算法
# 等
