import torch
import torch.nn as nn
from typing import List, Optional

# ============================================================
# 因子分解工具
# ============================================================

def find_factors_balanced(n: int, num_factors: int) -> List[int]:
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


def calculate_quantum_bonds(S_list: List[torch.Tensor], quantum_scale: float = 1.0, min_bond: int = 4, max_bond: int = 99999):
    """
    🌟 纯量子自组织截断引擎：不看任何人造预算，只遵从物理学 $2^S$ 法则。
    """
    dynamic_bonds = []
    entropies = []
    
    for S in S_list:
        # 1. 能量概率分布
        energy = S ** 2
        # 🛠️ 修复：绝对的非负钳制，防止 NaN
        prob = (energy / energy.sum().clamp(min=1e-12)).clamp(min=1e-12)
        
        # 2. 计算以 2 为底的香农熵 (Shannon Entropy)
        entropy = -torch.sum(prob * torch.log2(prob)).item()
        entropies.append(entropy)
        
        # 3. 💥 核心物理公式：\chi = \alpha * 2^S
        bond = int(round(quantum_scale * (2 ** entropy)))
        
        # 防止极端情况导致矩阵碎裂或爆显存
        bond = max(min_bond, min(max_bond, bond))
        dynamic_bonds.append(bond)
        
    return dynamic_bonds, entropies


def calculate_dynamic_bonds_by_entropy(S_list: List[torch.Tensor], target_total_bond: int, min_bond: int = 4) -> List[int]:
    """
    根据每个张量切面的奇异值纠缠熵，按比例分配总 Bond 预算。
    """
    entropies = []
    for S in S_list:
        energy = S ** 2
        # 🛠️ 修复：绝对的非负钳制，防止 NaN
        prob = (energy / energy.sum().clamp(min=1e-12)).clamp(min=1e-12)
        
        entropy = -torch.sum(prob * torch.log2(prob)).item()
        entropies.append(entropy)
        
    # 3. 预算分配
    total_entropy = sum(entropies) + 1e-9
    dynamic_bonds = []
    
    for entropy in entropies:
        weight = entropy / total_entropy
        bond = int(round(weight * target_total_bond))
        bond = max(min_bond, bond) # 保证物理骨架不断裂
        dynamic_bonds.append(bond)
        
    return dynamic_bonds, entropies


def factor_linear_mpo_custom(
    weight: torch.Tensor,
    bond_dim: int,              
    num_cores: int,
    out_fac: Optional[List[int]] = None,
    in_fac: Optional[List[int]] = None,
    s_vector: Optional[torch.Tensor] = None,
    boundary: str = "open",
    noise_scale: float = 1e-5,
    adaptive_mode: str = "quantum", # 默认设为 quantum
    energy_threshold: float = 0.99,
    quantum_scale: float = 2.0,     
    min_bond: int = 4,              
) -> List[torch.Tensor]:

    assert num_cores >= 2, "num_cores must be >= 2"

    W_orig = weight.detach().clone()
    orig_dtype = W_orig.dtype
    device = W_orig.device
    out_f, in_f = W_orig.shape

    if out_fac is None: out_fac = find_factors_balanced(out_f, num_cores)
    if in_fac is None: in_fac = find_factors_balanced(in_f, num_cores)

    def _execute_mpo_pass(W_tensor, custom_bonds=None, mode="fixed"):
        W = W_tensor.to(torch.float32)
        if s_vector is not None:
            s_vec = s_vector.to(device=device, dtype=torch.float32)
            W = W * s_vec.unsqueeze(0)

        if not torch.isfinite(W).all():
            W = torch.nan_to_num(W, nan=0.0, posinf=1e4, neginf=-1e4)

        T = W.reshape(*out_fac, *in_fac)
        perm = [j for i in range(num_cores) for j in (i, num_cores + i)]
        T = T.permute(*perm).contiguous()

        cores, prev = [], 1
        actual_ranks = []
        S_history = []

        for k in range(num_cores - 1):
            rows = prev * out_fac[k] * in_fac[k]
            T = T.reshape(rows, -1)

            scale = T.abs().amax()
            T_scaled = T / scale if float(scale) > 0 else T
            T_scaled = T_scaled.contiguous()
            
            try:
                Q, R = torch.linalg.qr(T_scaled, mode="reduced")
                R = R.contiguous()
                try:
                    U_r, S, Vh = torch.linalg.svd(R, full_matrices=False, driver="gesvd")
                except TypeError:
                    U_r, S, Vh = torch.linalg.svd(R, full_matrices=False)
            except Exception:
                T64 = T_scaled.detach().to(device="cpu", dtype=torch.float64)
                U_r, S, Vh = torch.linalg.svd(T64, full_matrices=False)
                U_r, S, Vh = U_r.to(T.device).float(), S.to(T.device).float(), Vh.to(T.device).float()
                Q = None

            U = U_r if Q is None else Q @ U_r
            S = S * scale
            
            S_history.append(S.clone())
            rank_avail = min(U.shape[1], S.shape[0], Vh.shape[0])

            if mode == "energy":
                S_sq = S[:rank_avail] ** 2
                total_energy = S_sq.sum().item()
                if total_energy > 0:
                    cumsum = torch.cumsum(S_sq, dim=0)
                    mask = cumsum >= energy_threshold * total_energy
                    r = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1 if mask.any() else rank_avail
                else:
                    r = 1
            elif mode == "custom_list" and custom_bonds is not None:
                r = custom_bonds[k]
            else: 
                r = bond_dim

            r = max(min_bond, min(int(r), int(rank_avail)))
            actual_ranks.append(r)

            U, S_trunc, Vh = U[:, :r], S[:r], Vh[:r]
            cores.append(U.reshape(prev, out_fac[k], in_fac[k], r))
            T = torch.diag(S_trunc) @ Vh
            prev = r

        cores.append(T.reshape(prev, out_fac[-1], in_fac[-1], 1))
        
        cleaned = []
        for c in cores:
            if not torch.isfinite(c).all(): c = torch.nan_to_num(c, nan=0.0)
            cleaned.append(c.to(device=device, dtype=orig_dtype))
            
        return cleaned, S_history, actual_ranks


    # ========================================================
    # 策略路由枢纽
    # ========================================================
    if adaptive_mode == "entropy":
        # 1. 探针 Pass：用无限制的 bond (99999) 探查矩阵的真实物理结构
        # 🌟 修复：必须用 custom_list，否则 99999 不会生效，会被外面的 bond_dim 假截断！
        _, S_list, _ = _execute_mpo_pass(W_orig, mode="custom_list", custom_bonds=[99999]*(num_cores-1))
        
        target_total = bond_dim * (num_cores - 1)
        allocated_bonds, entropies = calculate_dynamic_bonds_by_entropy(S_list, target_total, min_bond)
        print(f"    [Entropy 路由] 各切口纠缠熵: {[f'{e:.2f}' for e in entropies]}")
        print(f"    [Entropy 路由] 预算重新分配: {bond_dim}x{num_cores-1} -> {allocated_bonds}")
        cores, _, final_ranks = _execute_mpo_pass(W_orig, custom_bonds=allocated_bonds, mode="custom_list")
        
    elif adaptive_mode == "energy":
        cores, _, final_ranks = _execute_mpo_pass(W_orig, mode="energy")
        print(f"    [Energy 自适应] 动态截断秩: {final_ranks} (保底 {energy_threshold*100}% 能量)")
        
    elif adaptive_mode == "quantum":
        # 1. 探针 Pass：拿真实奇异值
        # 🌟 修复：同上！
        _, S_list, _ = _execute_mpo_pass(W_orig, mode="custom_list", custom_bonds=[99999]*(num_cores-1))
        print("quantum scale:", quantum_scale)
        
        allocated_bonds, entropies = calculate_quantum_bonds(
            S_list, 
            quantum_scale=quantum_scale, 
            min_bond=min_bond, 
            max_bond=bond_dim 
        )
        
        print(f"    [Quantum 路由] 测得纠缠熵 S: {[f'{e:.2f}' for e in entropies]}")
        print(f"    [Quantum 路由] 2^S 独立裁决 Bond: {allocated_bonds}")
        cores, _, final_ranks = _execute_mpo_pass(W_orig, custom_bonds=allocated_bonds, mode="custom_list")

    else: # "fixed"
        cores, _, final_ranks = _execute_mpo_pass(W_orig, mode="fixed")

    if boundary.lower() == "periodic":
        r_0 = int(final_ranks[0]) if len(final_ranks) > 0 else int(bond_dim)
        pad_dim = r_0 - 1
        if pad_dim > 0:
            c1, cK = cores[0], cores[-1]
            noise_c1 = torch.randn(pad_dim, c1.shape[1], c1.shape[2], c1.shape[3], device=device, dtype=orig_dtype) * noise_scale
            cores[0] = torch.cat([c1, noise_c1], dim=0)
            noise_cK = torch.randn(cK.shape[0], cK.shape[1], cK.shape[2], pad_dim, device=device, dtype=orig_dtype) * noise_scale
            cores[-1] = torch.cat([cK, noise_cK], dim=3)

    return cores


def estimate_bond_dim(weight: torch.Tensor, num_cores: int, target_ratio: float = 0.1) -> int:
    out_f, in_f = weight.shape
    dense_params = out_f * in_f
    out_fac = find_factors_balanced(out_f, num_cores)
    in_fac = find_factors_balanced(in_f, num_cores)
    if num_cores == 2:
        term = out_fac[0] * in_fac[0] + out_fac[1] * in_fac[1]
        bond = int(target_ratio * dense_params / term) if term > 0 else 1
    else:
        avg_oi = sum(o * i for o, i in zip(out_fac, in_fac)) / num_cores
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

def reconstruct_mpo_matrix(cores: List[torch.Tensor]) -> torch.Tensor:
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
    D_out, D_in = o1 * o2 * o3, i1 * i2 * i3

    print(f"=== 1. 初始化数据 ===")
    A = torch.randn(D_out, D_in)
    
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    S_new = torch.exp(-torch.arange(len(S))/10.0) * 1000 
    A = U @ torch.diag(S_new) @ V
    
    X = torch.randn(D_in)
    Y_gt = A @ X
    X_tensor = X.view(i1, i2, i3)

    print(f"=== 2. Entropy 模式验证 (总预算 bond_dim=30) ===")
    cores_entropy = factor_linear_mpo_custom(A, bond_dim=30, num_cores=3, adaptive_mode="entropy")
    Y_mpo_t = torch.einsum('aoib,bpjc,cqkd,ijk->aopqd', *cores_entropy, X_tensor)
    Y_mpo = Y_mpo_t.squeeze(0).squeeze(-1).reshape(-1)
    print(f"Max diff: {torch.max(torch.abs(Y_gt - Y_mpo)).item():.3e}\n")

    print(f"=== 3. Energy 模式验证 (threshold=0.99) ===")
    cores_energy = factor_linear_mpo_custom(A, bond_dim=30, num_cores=3, adaptive_mode="energy", energy_threshold=0.99)
    Y_mpo_t2 = torch.einsum('aoib,bpjc,cqkd,ijk->aopqd', *cores_energy, X_tensor)
    Y_mpo2 = Y_mpo_t2.squeeze(0).squeeze(-1).reshape(-1)
    print(f"Max diff: {torch.max(torch.abs(Y_gt - Y_mpo2)).item():.3e}\n")

    print(f"=== 4. Quantum 模式验证 (quantum_scale=2.5, 硬上限=64) ===")
    cores_quantum = factor_linear_mpo_custom(A, bond_dim=64, num_cores=3, adaptive_mode="quantum", quantum_scale=2.5)
    Y_mpo_t3 = torch.einsum('aoib,bpjc,cqkd,ijk->aopqd', *cores_quantum, X_tensor)
    Y_mpo3 = Y_mpo_t3.squeeze(0).squeeze(-1).reshape(-1)
    print(f"Max diff: {torch.max(torch.abs(Y_gt - Y_mpo3)).item():.3e}\n")

if __name__ == "__main__":
    verify_mpo_equivalence()