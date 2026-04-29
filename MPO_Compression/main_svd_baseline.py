import os
import sys
# 强制只在 0, 1, 2 号 GPU 上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
import torch.nn.functional as F
import argparse
import gc
import csv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.cluster.hierarchy import linkage, leaves_list
from pathlib import Path
from calibration import get_activation_scales

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from test_MPO import factor_linear_mpo_custom, reconstruct_mpo_matrix, estimate_bond_dim

# =====================================================================
# 🌟 全局梯度松弛 (VQE/DMRG-style) - 已修复 no_grad 冲突
# =====================================================================
def relax_mpo_cores_globally(W_target, cores, order="oi", steps=300, lr=2e-3):
    # 🛡️ 强行开启梯度图构建，无视外层的 with torch.no_grad()
    with torch.enable_grad():
        device = W_target.device
        W_target = W_target.detach().float()
        
        # 拷贝 cores 并作为叶子节点开启梯度
        opt_cores = [c.detach().clone().float().to(device) for c in cores]
        for c in opt_cores:
            c.requires_grad_(True)
            
        optimizer = torch.optim.AdamW(opt_cores, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        def _contract_fn(c_list):
            x = c_list[0]
            for c in c_list[1:]:
                x = torch.tensordot(x, c, dims=([-1], [0]))
            if x.shape[0] == 1: x = x[0]
            if x.shape[-1] == 1: x = x[..., 0]
            num = len(c_list)
            if order == "oi":
                perm = list(range(0, 2 * num, 2)) + list(range(1, 2 * num, 2))
                x = x.permute(*perm).contiguous()
                o_shape = x.shape[:num]
                i_shape = x.shape[num:]
                W = x.view(int(torch.tensor(o_shape).prod()), int(torch.tensor(i_shape).prod()))
            else:
                perm = list(range(1, 2 * num, 2)) + list(range(0, 2 * num, 2))
                x = x.permute(*perm).contiguous()
                i_shape = x.shape[:num]
                o_shape = x.shape[num:]
                W = x.view(int(torch.tensor(i_shape).prod()), int(torch.tensor(o_shape).prod())).t()
            return W

        # 开始寻优
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            W_pred = _contract_fn(opt_cores)
            loss = F.mse_loss(W_pred, W_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_cores, 1.0)
            optimizer.step()
            scheduler.step()

        # 寻优结束，脱离计算图返回
        return [c.detach() for c in opt_cores]

def find_permutation_idx(W_gate, W_up):
    features = torch.cat([W_gate, W_up], dim=1).detach().cpu().float().numpy()
    Z = linkage(features, method='ward', metric='euclidean')
    perm = torch.tensor(leaves_list(Z), dtype=torch.long)
    inv_perm = torch.argsort(perm) 
    return perm, inv_perm

def compress_matrix(W_orig, target_ratio, mode, perm=None, inv_perm=None, is_down_proj=False, lora_rank=32, num_cores=3, adaptive_mode="entropy", quantum_scale=2.5, energy_threshold=0.99, s_vector=None):
    bond_dim = 256 if adaptive_mode == "quantum" else estimate_bond_dim(W_orig, num_cores, target_ratio)
    
    if mode in [3, 4, 5, 6]:
        W_perm = W_orig[:, perm].contiguous() if is_down_proj else W_orig[perm, :].contiguous()
    else:
        W_perm = W_orig.contiguous()

    W_compute = W_perm.float()
    
    s_safe = None
    if s_vector is not None:
        s_safe = s_vector.float().clamp(min=s_vector.float().mean() * 0.05)
        if mode in [3, 4, 5, 6] and is_down_proj:
            s_safe = s_safe[perm].contiguous()

    try:
        cores = factor_linear_mpo_custom(
            W_compute, bond_dim=bond_dim, num_cores=num_cores,
            s_vector=s_safe, 
            adaptive_mode=adaptive_mode, quantum_scale=quantum_scale, energy_threshold=energy_threshold
        )
        
        # 🌟 如果启用了松弛模式 (5, 6)，进行全局寻优
        if mode in [5, 6]:
            W_scaled_target = W_compute * s_safe.unsqueeze(0) if s_safe is not None else W_compute
            cores = relax_mpo_cores_globally(W_scaled_target, cores, steps=300)

        W_mpo_recon_scaled = reconstruct_mpo_matrix(cores)

        if mode in [2, 4, 6]:
            W_scaled = W_compute * s_safe.unsqueeze(0) if s_safe is not None else W_compute
            Delta_W_scaled = W_scaled - W_mpo_recon_scaled
            U, S, Vh = torch.linalg.svd(Delta_W_scaled, full_matrices=False)

    except RuntimeError as e:
        W_cpu = W_compute.cpu()
        s_cpu = s_safe.cpu() if s_safe is not None else None
        cores_cpu = factor_linear_mpo_custom(
            W_cpu, bond_dim, num_cores, s_vector=s_cpu,
            adaptive_mode=adaptive_mode, quantum_scale=quantum_scale, energy_threshold=energy_threshold
        )
        
        if mode in [5, 6]:
            W_scaled_target_cpu = W_cpu * s_cpu.unsqueeze(0) if s_cpu is not None else W_cpu
            cores_cpu = relax_mpo_cores_globally(W_scaled_target_cpu.to(W_orig.device), [c.to(W_orig.device) for c in cores_cpu], steps=300)
            cores_cpu = [c.cpu() for c in cores_cpu]

        W_mpo_recon_scaled_cpu = reconstruct_mpo_matrix(cores_cpu)
        W_mpo_recon_scaled = W_mpo_recon_scaled_cpu.to(W_orig.device)

        if mode in [2, 4, 6]:
            W_scaled_cpu = W_cpu * s_cpu.unsqueeze(0) if s_cpu is not None else W_cpu
            Delta_W_scaled_cpu = W_scaled_cpu - W_mpo_recon_scaled_cpu
            U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(Delta_W_scaled_cpu, full_matrices=False)
            U, S, Vh = U_cpu.to(W_orig.device), S_cpu.to(W_orig.device), Vh_cpu.to(W_orig.device)

    if mode in [2, 4, 6]:
        S_clamped = S[:lora_rank].clamp(min=0)
        A = U[:, :lora_rank] @ torch.diag(torch.sqrt(S_clamped))
        B = torch.diag(torch.sqrt(S_clamped)) @ Vh[:lora_rank, :]
        W_approx_scaled = W_mpo_recon_scaled + (A @ B)
    else:
        W_approx_scaled = W_mpo_recon_scaled

    if s_safe is not None:
        W_approx = W_approx_scaled / s_safe.unsqueeze(0)
    else:
        W_approx = W_approx_scaled

    W_approx = W_approx.to(W_orig.dtype)

    if mode in [3, 4, 5, 6]:
        W_final = W_approx[:, inv_perm].contiguous() if is_down_proj else W_approx[inv_perm, :].contiguous()
    else:
        W_final = W_approx

    return W_final

def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=10000):
    model.eval() 
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    nlls = []
    for begin in tqdm(range(0, seq_len, stride), desc="评测 PPL", leave=False):
        end = min(begin + max_len, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1: continue
        with torch.no_grad():
            out = model(chunk)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), reduction="mean"
            )
            nlls.append(loss.item())

    return round(torch.exp(torch.tensor(nlls).mean()).item(), 2)

def append_to_csv(filename, mode_name, ppl):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Mode', 'PPL'])
        writer.writerow([mode_name, ppl])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--adaptive_mode", type=str, default="entropy", choices=["fixed", "energy", "entropy", "quantum"])
    parser.add_argument("--quantum_scale", type=float, default=2.5)
    parser.add_argument("--energy_threshold", type=float, default=0.99)
    parser.add_argument("--csv_name", type=str, default="joint_6layer_ablation.csv")
    args = parser.parse_args()

    print("="*70)
    print(" 🔬 多层联合消融测试 (Multi-Layer Joint Ablation)")
    print(" ⚠️ 逻辑: 每次将 6 个核心层同时替换为压缩矩阵，测试误差级联爆炸！")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    baseline_ppl = eval_ppl(model, tokenizer)
    print(f"\n🌟 原版 Baseline PPL: {baseline_ppl}")
    append_to_csv(args.csv_name, "Baseline (Full Model)", baseline_ppl)

    print("\n🔍 提取 ASVD 分布...")
    activation_scales_dict = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    
    modes = [
        #(1, "Mode 1: Pure MPO"),
        #(2, "Mode 2: MPO + LoRA"),
        #(3, "Mode 3: Perm + MPO"),
        #(4, "Mode 4: Perm + MPO + LoRA"),
        (5, "Mode 5: Perm + MPO (Relaxed)"),         
        (6, "Mode 6: Perm + Relaxed MPO + LoRA")     
    ]

    test_layers = [10, 15, 20, 24, 28, 31]
    
    # 💾 Step 1: 提前把 6 层的原始权重备份到内存，避免反复从磁盘载入
    print(f"\n💾 正在备份测试层 {test_layers} 的原始权重...")
    orig_weights = {}
    for l_idx in test_layers:
        target_mlp = model.model.layers[l_idx].mlp
        orig_weights[l_idx] = {
            "gate": target_mlp.gate_proj.weight.data.clone(),
            "up":   target_mlp.up_proj.weight.data.clone(),
            "down": target_mlp.down_proj.weight.data.clone()
        }

    # 🚀 Step 2: 遍历不同模式，同时破坏 6 层
    for mode_idx, mode_name in modes:
        print("\n" + "="*60)
        print(f" 🧪 正在测试: {mode_name}")
        print("="*60)
        
        # 同时对 6 层注入当前 Mode 的压缩矩阵
        for l_idx in tqdm(test_layers, desc=f"注入 {mode_name} 至各层", leave=False):
            target_mlp = model.model.layers[l_idx].mlp
            orig = orig_weights[l_idx]
            
            s_vec_gate = activation_scales_dict.get(f"model.layers.{l_idx}.mlp.gate_proj")
            s_vec_up   = activation_scales_dict.get(f"model.layers.{l_idx}.mlp.up_proj")
            s_vec_down = activation_scales_dict.get(f"model.layers.{l_idx}.mlp.down_proj")

            # 计算当前层的 Permutation
            W_gate_scaled = orig["gate"].float() * (s_vec_gate.to(orig["gate"].device).float() if s_vec_gate is not None else 1.0)
            W_up_scaled   = orig["up"].float()   * (s_vec_up.to(orig["up"].device).float()     if s_vec_up is not None else 1.0)
            perm, inv_perm = find_permutation_idx(W_gate_scaled, W_up_scaled)

            with torch.no_grad():
                target_mlp.gate_proj.weight.data = compress_matrix(
                    orig["gate"], args.ratio, mode_idx, perm, inv_perm, is_down_proj=False,
                    adaptive_mode=args.adaptive_mode, quantum_scale=args.quantum_scale, energy_threshold=args.energy_threshold,
                    s_vector=s_vec_gate
                )
                target_mlp.up_proj.weight.data = compress_matrix(
                    orig["up"], args.ratio, mode_idx, perm, inv_perm, is_down_proj=False,
                    adaptive_mode=args.adaptive_mode, quantum_scale=args.quantum_scale, energy_threshold=args.energy_threshold,
                    s_vector=s_vec_up
                )
                target_mlp.down_proj.weight.data = compress_matrix(
                    orig["down"], args.ratio, mode_idx, perm, inv_perm, is_down_proj=True,
                    adaptive_mode=args.adaptive_mode, quantum_scale=args.quantum_scale, energy_threshold=args.energy_threshold,
                    s_vector=s_vec_down
                )
                
        # 测一次整体 PPL（此时 6 层同时被压缩）
        ppl = eval_ppl(model, tokenizer)
        print(f" 📈 联合截断后 PPL: {ppl}")
        append_to_csv(args.csv_name, mode_name, ppl)
        
        # 🛡️ Step 3: 将这 6 层满血复活，准备下一个 Mode 的测试
        print(" 🔄 恢复满血状态...")
        with torch.no_grad():
            for l_idx in test_layers:
                target_mlp = model.model.layers[l_idx].mlp
                target_mlp.gate_proj.weight.data.copy_(orig_weights[l_idx]["gate"])
                target_mlp.up_proj.weight.data.copy_(orig_weights[l_idx]["up"])
                target_mlp.down_proj.weight.data.copy_(orig_weights[l_idx]["down"])
                
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n🎉 联合消融测试完成！结果已保存在 {args.csv_name}")

if __name__ == "__main__":
    main()
    # 原模型的PPL：6.06，Mode 1: 11.78，Mode 2: 9.07，Mode 3: 10.37，Mode 4: 9.03，Mode 5: 10.38，Mode 6: 9.05