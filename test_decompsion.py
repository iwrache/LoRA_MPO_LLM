import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
import math
import os

# 激活 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PermutationAwareMPO(nn.Module):
    def __init__(self, out_dims: list, in_dims: list, bond_dim: int, row_perm: torch.Tensor, col_perm: torch.Tensor):
        super().__init__()
        assert len(out_dims) == len(in_dims), "Site 数量不一致"
        
        self.out_dims = out_dims
        self.in_dims = in_dims
        self.num_sites = len(out_dims)
        self.bond_dim = bond_dim
        self.out_features = math.prod(out_dims)
        self.in_features = math.prod(in_dims)

        self.register_buffer('row_perm', row_perm)
        self.register_buffer('inv_col_perm', torch.argsort(col_perm))
        self.cores = nn.ParameterList()

    @torch.no_grad()
    def from_matrix(self, W_wild: torch.Tensor):
        # 反解重排
        inv_row_perm = torch.argsort(self.row_perm)
        W_clean = W_wild[inv_row_perm][:, self.inv_col_perm]
        
        # 张量化与交织
        W_tensor = W_clean.reshape(*self.out_dims, *self.in_dims)
        perm = []
        for i in range(self.num_sites):
            perm.extend([i, self.num_sites + i])
        W_tensor = W_tensor.permute(*perm).contiguous()
        site_dims = [self.out_dims[i] * self.in_dims[i] for i in range(self.num_sites)]
        W_tensor = W_tensor.reshape(*site_dims)
        
        # TT-SVD 截断
        current_tensor = W_tensor
        r_prev = 1
        cores = []
        for k in range(self.num_sites - 1):
            dim_k = site_dims[k]
            current_tensor = current_tensor.reshape(r_prev * dim_k, -1)
            U, S, Vh = linalg.svd(current_tensor, full_matrices=False)
            r_k = min(self.bond_dim, S.shape[0])
            U, S, Vh = U[:, :r_k], S[:r_k], Vh[:r_k, :]
            
            core = U.reshape(r_prev, self.out_dims[k], self.in_dims[k], r_k)
            cores.append(nn.Parameter(core))
            current_tensor = torch.diag(S) @ Vh
            r_prev = r_k
            
        core_last = current_tensor.reshape(r_prev, self.out_dims[-1], self.in_dims[-1], 1)
        cores.append(nn.Parameter(core_last))
        self.cores = nn.ParameterList(cores)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[0]
        T = X[:, self.inv_col_perm]
        for k in range(self.num_sites):
            C = self.cores[k]
            R_in, O_k, I_k, R_out = C.shape
            right_size = math.prod(self.in_dims[k+1:]) if k+1 < self.num_sites else 1
            left_size = T.numel() // (R_in * I_k * right_size)
            T = T.reshape(left_size, R_in, I_k, right_size)
            T = torch.einsum('lrig, rois -> losg', T, C)
        Y_clean = T.reshape(batch_size, self.out_features)
        Y_wild = Y_clean[:, self.row_perm]
        return Y_wild

    @torch.no_grad()
    def reconstruct_wild_matrix(self):
        res = self.cores[0]
        for k in range(1, self.num_sites):
            res = torch.tensordot(res, self.cores[k], dims=([-1], [0]))
        flat_dims = []
        for o, i in zip(self.out_dims, self.in_dims):
            flat_dims.extend([o, i])
        res = res.reshape(*flat_dims)
        inv_perm = [2*i for i in range(self.num_sites)] + [2*i + 1 for i in range(self.num_sites)]
        res = res.permute(*inv_perm).contiguous()
        W_clean_approx = res.reshape(self.out_features, self.in_features)
        
        inv_col_perm = torch.argsort(self.inv_col_perm)
        return W_clean_approx[self.row_perm][:, inv_col_perm]

    @torch.no_grad()
    def evaluate_errors(self, W_target: torch.Tensor, X_input: torch.Tensor):
        W_approx = self.reconstruct_wild_matrix()
        recon_err = torch.norm(W_target - W_approx) / torch.norm(W_target)
        Y_true = F.linear(X_input, W_target)
        Y_mpo = self.forward(X_input)
        forward_err = torch.norm(Y_true - Y_mpo) / torch.norm(Y_true)
        return recon_err.item(), forward_err.item()

# =============================================================================
# 🔬 三足鼎立验证实验：SVD vs Naive MPO vs Permutation-Aware MPO
# =============================================================================
def test_three_way_comparison():
    print("="*80)
    print("🔬 终极对照实验：SVD vs Naive MPO vs Permutation-Aware MPO")
    print("="*80)
    torch.manual_seed(42)

    out_dims, in_dims = [4, 4, 4, 4], [4, 4, 4, 4]
    dim_out, dim_in = 256, 256

    # 1. 制造高维张量结构的矩阵 (在 4D 空间内 Rank 极低，但在 2D 空间 Rank 高达 16)
    A, B, C, D = [torch.randn(4, 4, device=device) for _ in range(4)]
    W_clean = torch.kron(torch.kron(torch.kron(A, B), C), D)

    # 2. 制造全局乱序野生矩阵
    row_perm = torch.randperm(dim_out, device=device)
    col_perm = torch.randperm(dim_in, device=device)
    W_wild = W_clean[row_perm][:, col_perm]
    X = torch.randn(32, dim_in, device=device)
    Y_true = F.linear(X, W_wild)

    # --------------------------------------------------
    # 🌟 选手 A：我们的 SOTA 方案 (Permutation-Aware MPO)
    # --------------------------------------------------
    bond_dim = 2
    sota_mpo = PermutationAwareMPO(out_dims, in_dims, bond_dim, row_perm, col_perm).to(device)
    sota_mpo.from_matrix(W_wild)
    num_params_mpo = sum(p.numel() for p in sota_mpo.parameters())
    sota_recon, sota_fwd = sota_mpo.evaluate_errors(W_wild, X)

    # --------------------------------------------------
    # 🤡 选手 B：天真的直切 MPO (无排列重组)
    # --------------------------------------------------
    # 传入恒等排列 (0,1,2...255)，模拟不对权重做重排，直接硬压 W_wild
    id_perm = torch.arange(dim_out, device=device)
    naive_mpo = PermutationAwareMPO(out_dims, in_dims, bond_dim, id_perm, id_perm).to(device)
    naive_mpo.from_matrix(W_wild)
    naive_recon, naive_fwd = naive_mpo.evaluate_errors(W_wild, X)

    # --------------------------------------------------
    # 🧱 选手 C：传统 SVD (分配同样的，甚至更多的参数量)
    # --------------------------------------------------
    # 计算能分配给 SVD 的最大 Rank
    r_svd = max(1, num_params_mpo // (dim_out + dim_in)) 
    # 即使 Rank=1，参数量为 1*(256+256)=512，也已经大于 MPO 的参数量(192)了！
    
    U, S, Vh = linalg.svd(W_wild, full_matrices=False)
    W_svd = U[:, :r_svd] @ torch.diag(S[:r_svd]) @ Vh[:r_svd, :]
    Y_svd = F.linear(X, W_svd)
    
    svd_recon = (torch.norm(W_wild - W_svd) / torch.norm(W_wild)).item()
    svd_fwd = (torch.norm(Y_true - Y_svd) / torch.norm(Y_true)).item()

    # --------------------------------------------------
    # 📊 输出报告
    # --------------------------------------------------
    num_params_orig = dim_out * dim_in
    num_params_svd = r_svd * (dim_out + dim_in)

    print(f"📦 参数量预算底线: ~{num_params_mpo} (压缩率极高: <0.5%)")
    print(f"  - 原矩阵参数量: {num_params_orig}")
    print(f"  - SVD 分配参数: {num_params_svd} (Rank={r_svd}) <- 注意，SVD已经占便宜了！")
    print(f"  - MPO 分配参数: {num_params_mpo} (Bond={bond_dim})\n")

    print(f"{'模型方案':<30} | {'相对重建误差':<15} | {'前向输出误差':<15}")
    print("-" * 65)
    print(f"{'[基准 1] 传统 SVD':<26} | {svd_recon:<20.4f} | {svd_fwd:<15.4f}")
    print(f"{'[基准 2] Naive MPO (无重排)':<21} | {naive_recon:<20.4f} | {naive_fwd:<15.4f}")
    print(f"{'👑 [我们的] Permutation MPO':<22} | {sota_recon:<20.8f} | {sota_fwd:<15.8f}")
    print("="*80)

if __name__ == "__main__":
    test_three_way_comparison()