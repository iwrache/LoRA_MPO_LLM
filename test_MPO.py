import torch
import torch.nn as nn
from typing import List, Optional

# ============================================================
# 因子分解工具（自包含版，兼容项目代码逻辑）
# ============================================================

def find_factors_balanced(n: int, num_factors: int) -> List[int]:
    """
    将 n 分解为 num_factors 个乘积因子，尽量平衡。
    直接复制自项目 helpers.py，保证 test_MPO.py 可独立运行。
    """
    if num_factors == 1:
        return [int(n)]
    factors = []
    d = 2
    temp_n = int(n)
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors.append(d)
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors.append(temp_n)
    groups = [1] * num_factors
    for factor in sorted(factors, reverse=True):
        groups.sort()
        groups[0] *= factor
    return [int(g) for g in groups]


# ============================================================
# 核心：自定义 MPO 分解（带 bond_dim 截断）
# ============================================================

def factor_linear_mpo_custom(
    weight: torch.Tensor,
    bond_dim: int,              # 硬上限，防止爆显存
    num_cores: int,
    out_fac: Optional[List[int]] = None,
    in_fac: Optional[List[int]] = None,
    s_vector: Optional[torch.Tensor] = None,
    boundary: str = "open",
    noise_scale: float = 1e-5,
    # ====== 新增参数 ======
    adaptive: bool = True,     # 是否启用自适应截断
    energy_threshold: float = 0.99,  # 保留的能量比例（如 99.9%）
    min_bond: int = 4,          # 自适应模式下的最小 bond，防止退化
) -> List[torch.Tensor]:
    """
    将权重矩阵 W (out_f, in_f) 用 sequential SVD 分解为 MPO cores。

    两种截断策略：
      - adaptive=False（默认）: 所有步统一截断到 bond_dim，与原来行为一致。
      - adaptive=True:  每步根据奇异值能量阈值动态决定截断秩，
                        同时受 bond_dim 硬上限和 min_bond 下限约束。
    """
    assert num_cores >= 2, "num_cores must be >= 2"

    W = weight.detach().clone()
    orig_dtype = W.dtype
    device = weight.device
    out_f, in_f = W.shape
    W = W.to(torch.float32)

    if s_vector is not None:
        s_vector = s_vector.to(device=device, dtype=torch.float32)
        W = W * s_vector.unsqueeze(0)

    if not torch.isfinite(W).all():
        W = torch.nan_to_num(W, nan=0.0, posinf=1e4, neginf=-1e4)

    if out_fac is None:
        out_fac = find_factors_balanced(out_f, num_cores)
    if in_fac is None:
        in_fac = find_factors_balanced(in_f, num_cores)

    T = W.reshape(*out_fac, *in_fac)
    del W
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    perm = [j for i in range(num_cores) for j in (i, num_cores + i)]
    T = T.permute(*perm).contiguous()

    cores, prev = [], 1
    adaptive_ranks = []  # 记录每步实际选择的秩（方便调试）

    for k in range(num_cores - 1):
        rows = prev * out_fac[k] * in_fac[k]
        T = T.reshape(rows, -1)

        scale = T.abs().amax()
        T_scaled = T / scale if float(scale) > 0 else T

        # QR + SVD
        try:
            Q, R = torch.linalg.qr(T_scaled, mode="reduced")
            try:
                U_r, S, Vh = torch.linalg.svd(R, full_matrices=False, driver="gesvd")
            except TypeError:
                U_r, S, Vh = torch.linalg.svd(R, full_matrices=False)
        except Exception:
            T64 = T_scaled.detach().to(device="cpu", dtype=torch.float64)
            U_r, S, Vh = torch.linalg.svd(T64, full_matrices=False)
            U_r = U_r.to(device=T.device, dtype=T.dtype)
            S = S.to(device=T.device, dtype=T.dtype)
            Vh = Vh.to(device=T.device, dtype=T.dtype)
            Q = None

        U = U_r if Q is None else Q @ U_r
        S = S * scale

        # ====== 截断策略分支 ======
        rank_avail = min(U.shape[1], S.shape[0], Vh.shape[0])

        if adaptive:
            # 基于能量阈值的自适应截断
            S_sq = S[:rank_avail] ** 2
            total_energy = S_sq.sum().item()

            if total_energy > 0:
                cumsum = torch.cumsum(S_sq, dim=0)
                # 找到满足能量阈值的最小秩
                mask = cumsum >= energy_threshold * total_energy
                if mask.any():
                    r_energy = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
                else:
                    r_energy = int(rank_avail)
            else:
                r_energy = 1

            # 施加上下限约束
            r = max(min_bond, min(r_energy, bond_dim, rank_avail))
        else:
            # 原始固定截断
            r = max(1, min(int(bond_dim), int(rank_avail)))

        adaptive_ranks.append(r)

        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r]

        core = U.reshape(prev, out_fac[k], in_fac[k], r)
        cores.append(core)

        del U, U_r, Q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        T = torch.diag(S) @ Vh

        del S, Vh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prev = r

    cores.append(T.reshape(prev, out_fac[-1], in_fac[-1], 1))
    del T
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 打印自适应截断的实际秩（调试用）
    if adaptive:
        print(f"    自适应截断秩: {adaptive_ranks}  (阈值={energy_threshold}, 上限={bond_dim})")

    # 清洗 NaN/Inf
    cleaned = []
    for c in cores:
        if not torch.isfinite(c).all():
            c = torch.nan_to_num(c, nan=0.0, posinf=1e4, neginf=-1e4)
        cleaned.append(c.to(device=device, dtype=orig_dtype))
    del cores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PBC 转换
    if boundary.lower() == "periodic":
        r_0 = int(bond_dim)
        pad_dim = r_0 - 1

        if pad_dim > 0:
            c1 = cleaned[0]
            cK = cleaned[-1]

            noise_c1 = torch.randn(pad_dim, c1.shape[1], c1.shape[2], c1.shape[3],
                                   device=device, dtype=orig_dtype) * noise_scale
            cleaned[0] = torch.cat([c1, noise_c1], dim=0)

            noise_cK = torch.randn(cK.shape[0], cK.shape[1], cK.shape[2], pad_dim,
                                   device=device, dtype=orig_dtype) * noise_scale
            cleaned[-1] = torch.cat([cK, noise_cK], dim=3)

    return cleaned


def estimate_bond_dim(weight: torch.Tensor, num_cores: int, target_ratio: float = 0.1) -> int:
    """
    根据目标压缩比自动估算 bond_dim。

    Args:
        weight:     [out_f, in_f]
        num_cores:  K
        target_ratio: MPO 参数量 / Dense 参数量的目标比例（例如 0.1 表示 10%）

    Returns:
        估算的 bond_dim（>=1）
    """
    out_f, in_f = weight.shape
    dense_params = out_f * in_f
    out_fac = find_factors_balanced(out_f, num_cores)
    in_fac = find_factors_balanced(in_f, num_cores)

    # MPO params 的近似公式
    if num_cores == 2:
        # core0: 1*o1*i1*chi + core1: chi*o2*i2*1 = chi * (o1*i1 + o2*i2)
        term = out_fac[0] * in_fac[0] + out_fac[1] * in_fac[1]
        bond = int(target_ratio * dense_params / term) if term > 0 else 1
    else:
        # 主导项近似: (K-2) * chi^2 * avg(o_k * i_k) + 2 * chi * end_term
        # 用迭代求解更稳妥
        avg_oi = sum(o * i for o, i in zip(out_fac, in_fac)) / num_cores
        # 直接解二次方程近似: (K-2)*avg_oi*chi^2 + 2*avg_oi*chi ≈ target * dense_params
        a = (num_cores - 2) * avg_oi
        b = 2 * avg_oi
        c_val = -target_ratio * dense_params
        if a > 0:
            bond = int((-b + (b * b - 4 * a * c_val) ** 0.5) / (2 * a))
        elif b > 0:
            bond = int(-c_val / b)
        else:
            bond = 1
    return max(1, bond)


# ============================================================
# 本地验证脚本
# ============================================================

def reconstruct_mpo_matrix(cores: List[torch.Tensor]) -> torch.Tensor:
    """将 MPO cores 重构为原始矩阵 W (out_f, in_f)。"""
    M = cores[0]
    for k in range(1, len(cores)):
        nd = M.ndim
        lhs1 = ''.join(chr(ord('a') + i) for i in range(nd - 1))
        contracted = chr(ord('a') + nd - 1)
        lhs2 = contracted + ''.join(chr(ord('a') + nd + i) for i in range(cores[k].ndim - 1))
        rhs = lhs1 + lhs2[1:]
        eq = f"{lhs1}{contracted},{lhs2}->{rhs}"
        M = torch.einsum(eq, M, cores[k])
    M = M.squeeze(0).squeeze(-1)
    K = len(cores)
    out_fac = [cores[k].shape[1] for k in range(K)]
    in_fac = [cores[k].shape[2] for k in range(K)]
    perm = [2 * i for i in range(K)] + [2 * i + 1 for i in range(K)]
    M = M.permute(*perm)
    return M.reshape(torch.prod(torch.tensor(out_fac)), torch.prod(torch.tensor(in_fac)))


def verify_mpo_equivalence():
    torch.set_default_dtype(torch.float64)

    o1, o2, o3 = 16, 16, 16
    i1, i2, i3 = 16, 16, 16
    D_out = o1 * o2 * o3
    D_in = i1 * i2 * i3

    print(f"=== 1. 初始化数据 ===")
    A = torch.randn(D_out, D_in)
    X = torch.randn(D_in)
    Y_gt = A @ X
    print(f"原始矩阵形状: {A.shape}")
    print(f"输入向量形状: {X.shape}")
    print(f"Dense 参数量: {D_out * D_in}\n")

    X_tensor = X.view(i1, i2, i3)

    # --- 满秩验证 ---
    print(f"=== 2. 满秩 MPO 验证 (bond_dim=999999) ===")
    cores_full = factor_linear_mpo_custom(A, bond_dim=999999, num_cores=3)
    Y_mpo_t = torch.einsum('aoib,bpjc,cqkd,ijk->aopqd', *cores_full, X_tensor)
    Y_mpo_t = Y_mpo_t.squeeze(0).squeeze(-1)
    Y_mpo = Y_mpo_t.reshape(-1)
    max_diff_full = torch.max(torch.abs(Y_gt - Y_mpo)).item()
    print(f"Max diff: {max_diff_full:.3e}")
    A_rec_full = reconstruct_mpo_matrix(cores_full)
    print(f"||A||_F: {A.norm('fro').item():.3f}  ||A_rec||_F: {A_rec_full.norm('fro').item():.3f}  rel_err: {(A - A_rec_full).norm('fro').item() / A.norm('fro').item():.3e}")
    print("✅ 满秩验证通过\n" if max_diff_full < 1e-4 else "❌ 满秩验证失败\n")

    # --- 截断验证 ---
    bond = 30
    print(f"=== 3. 截断 MPO 验证 (bond_dim={bond}) ===")
    cores_trunc = factor_linear_mpo_custom(A, bond_dim=bond, num_cores=3)
    Y_mpo_t2 = torch.einsum('aoib,bpjc,cqkd,ijk->aopqd', *cores_trunc, X_tensor)
    Y_mpo_t2 = Y_mpo_t2.squeeze(0).squeeze(-1)
    Y_mpo2 = Y_mpo_t2.reshape(-1)
    mpo_params = sum(c.numel() for c in cores_trunc)
    dense_params = D_out * D_in
    max_diff_trunc = torch.max(torch.abs(Y_gt - Y_mpo2)).item()
    print(f"Max diff: {max_diff_trunc:.3e}")
    A_rec_trunc = reconstruct_mpo_matrix(cores_trunc)
    print(f"||A||_F: {A.norm('fro').item():.3f}  ||A_rec_trunc||_F: {A_rec_trunc.norm('fro').item():.3f}  rel_err: {(A - A_rec_trunc).norm('fro').item() / A.norm('fro').item():.3e}")
    print(f"MPO 参数量: {mpo_params}")
    print(f"压缩比: {mpo_params / dense_params:.2%}\n")

    # --- 按比例自动估算 bond_dim ---
    target = 0.1
    print(f"=== 4. 按比例估算 bond_dim (target_ratio={target}) ===")
    est_bond = estimate_bond_dim(A, num_cores=3, target_ratio=target)
    print(f"估算 bond_dim: {est_bond}")
    cores_est = factor_linear_mpo_custom(A, bond_dim=est_bond, num_cores=3)
    mpo_params_est = sum(c.numel() for c in cores_est)
    actual_ratio = mpo_params_est / dense_params
    print(f"实际 MPO 参数量: {mpo_params_est}")
    print(f"实际压缩比: {actual_ratio:.2%}")
    print("✅ 估算成功\n" if abs(actual_ratio - target) / target < 0.5 else "⚠️ 估算偏差较大\n")


if __name__ == "__main__":
    verify_mpo_equivalence()
