import os
import sys
# 强制只在 0, 1, 2 号 GPU 上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
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

# 导入你写好的 MPO 核心工具
from test_MPO import factor_linear_mpo_custom, reconstruct_mpo_matrix, estimate_bond_dim

def find_permutation_idx(W_gate, W_up):
    """计算 Gate 和 Up 联合特征的最优聚类排列索引"""
    features = torch.cat([W_gate, W_up], dim=1).detach().cpu().float().numpy()
    Z = linkage(features, method='ward', metric='euclidean')
    perm = torch.tensor(leaves_list(Z), dtype=torch.long)
    inv_perm = torch.argsort(perm) # 逆排列，用于无损还原
    return perm, inv_perm

def compress_matrix(W_orig, target_ratio, mode, perm=None, inv_perm=None, is_down_proj=False, lora_rank=32, num_cores=3, adaptive_mode="entropy", quantum_scale=2.5, energy_threshold=0.99, s_vector=None):
    """
    包含 ASVD 缩放与逆缩放的终极压缩矩阵函数
    """
    bond_dim = 256 if adaptive_mode == "quantum" else estimate_bond_dim(W_orig, num_cores, target_ratio)
    
    # [Step A: 物理重排 (仅限模式 3, 4)]
    if mode in [3, 4]:
        W_perm = W_orig[:, perm].contiguous() if is_down_proj else W_orig[perm, :].contiguous()
    else:
        W_perm = W_orig.contiguous()

    W_compute = W_perm.float()
    
    # 🌟 ASVD 护盾准备：处理安全的 s_vector
    s_safe = None
    if s_vector is not None:
        s_safe = s_vector.float().clamp(min=s_vector.float().mean() * 0.05)
        if mode in [3, 4] and is_down_proj:
            # 如果是 down_proj，输入列被打乱，激活因子也必须同步打乱！
            s_safe = s_safe[perm].contiguous()

    try:
        # ⚡ GPU 光速 SVD 分解 (底层会自动使用 s_vector 进行加权)
        cores = factor_linear_mpo_custom(
            W_compute, bond_dim=bond_dim, num_cores=num_cores,
            s_vector=s_safe,  # 传入 ASVD 权重
            adaptive_mode=adaptive_mode, quantum_scale=quantum_scale, energy_threshold=energy_threshold
        )
        W_mpo_recon_scaled = reconstruct_mpo_matrix(cores)

        if mode in [2, 4]:
            # 计算残差时，必须用加权后的原始矩阵去减
            W_scaled = W_compute * s_safe.unsqueeze(0) if s_safe is not None else W_compute
            Delta_W_scaled = W_scaled - W_mpo_recon_scaled
            U, S, Vh = torch.linalg.svd(Delta_W_scaled, full_matrices=False)

    except RuntimeError as e:
        # 🛡️ CPU 降级保命
        W_cpu = W_compute.cpu()
        s_cpu = s_safe.cpu() if s_safe is not None else None
        cores_cpu = factor_linear_mpo_custom(
            W_cpu, bond_dim, num_cores, s_vector=s_cpu,
            adaptive_mode=adaptive_mode, quantum_scale=quantum_scale, energy_threshold=energy_threshold
        )
        W_mpo_recon_scaled_cpu = reconstruct_mpo_matrix(cores_cpu)
        W_mpo_recon_scaled = W_mpo_recon_scaled_cpu.to(W_orig.device)

        if mode in [2, 4]:
            W_scaled_cpu = W_cpu * s_cpu.unsqueeze(0) if s_cpu is not None else W_cpu
            Delta_W_scaled_cpu = W_scaled_cpu - W_mpo_recon_scaled_cpu
            U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(Delta_W_scaled_cpu, full_matrices=False)
            U, S, Vh = U_cpu.to(W_orig.device), S_cpu.to(W_orig.device), Vh_cpu.to(W_orig.device)

    # [Step C: 加上 LoRA 残差，并进行逆向缩放]
    if mode in [2, 4]:
        S_clamped = S[:lora_rank].clamp(min=0)
        A = U[:, :lora_rank] @ torch.diag(torch.sqrt(S_clamped))
        B = torch.diag(torch.sqrt(S_clamped)) @ Vh[:lora_rank, :]
        W_approx_scaled = W_mpo_recon_scaled + (A @ B)
    else:
        W_approx_scaled = W_mpo_recon_scaled

    # 🌟 核心 ASVD 逆缩放 (Unscaling)：因为这是单层替换，必须除以 s_safe 变回原来的刻度
    if s_safe is not None:
        W_approx = W_approx_scaled / s_safe.unsqueeze(0)
    else:
        W_approx = W_approx_scaled

    W_approx = W_approx.to(W_orig.dtype)

    # [Step D: 逆向重排还原 (仅限模式 3, 4)]
    if mode in [3, 4]:
        W_final = W_approx[:, inv_perm].contiguous() if is_down_proj else W_approx[inv_perm, :].contiguous()
    else:
        W_final = W_approx

    return W_final

def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=10000):
    """计算 WikiText-2 PPL (截取前10000词，以实现极速评测)"""
    model.eval() 
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    nlls = []
    for begin in range(0, seq_len, stride):
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

def append_to_csv(filename, layer_idx, mode_name, ppl):
    """将结果实时写入 CSV，防止程序中途中断丢失数据"""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Layer', 'Mode', 'PPL'])
        writer.writerow([layer_idx, mode_name, ppl])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--ratio", type=float, default=0.6, help="统一截断率 (Entropy模式用)")
    parser.add_argument("--adaptive_mode", type=str, default="entropy", choices=["fixed", "energy", "entropy", "quantum"])
    parser.add_argument("--quantum_scale", type=float, default=2.5)
    parser.add_argument("--energy_threshold", type=float, default=0.99)
    parser.add_argument("--csv_name", type=str, default="layerwise_ablation_results.csv")
    args = parser.parse_args()

    print("="*70)
    print(" 🔬 全层独立消融遍历实验 (Single-Layer Non-Cumulative Ablation)")
    print(" ⚠️ 逻辑: 每次仅压缩当前层，测完立刻恢复满血！")
    print("="*70)

    print("📦 加载模型 (BFloat16以节省显存)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    print("\n[Baseline] 计算满血原版 PPL...")
    baseline_ppl = eval_ppl(model, tokenizer)
    print(f"🌟 原版 PPL: {baseline_ppl}")
    append_to_csv(args.csv_name, "Baseline", "Full_Model", baseline_ppl)

    # =====================================================================
    # 🌟 新增：极速提取 ASVD 激活比例 (Activation Scales)
    # =====================================================================
    print("\n🔍 正在通过 32 条样本提取 Hessian-Aware ASVD 激活分布...")
    activation_scales_dict = get_activation_scales(model, tokenizer, num_samples=32, max_len=256)
    
    num_layers = len(model.model.layers)
    
    modes = [
        (1, "Mode 1: Pure MPO"),
        (2, "Mode 2: MPO + LoRA"),
        (3, "Mode 3: Perm + MPO"),
        (4, "Mode 4: Perm + MPO + LoRA")
    ]

    # 🚀 遍历模型的每一层
    for layer_idx in range(27, num_layers):
        print("\n" + "="*50)
        print(f" 🎯 正在独立测试 Layer {layer_idx} / {num_layers - 1}")
        print("="*50)
        
        target_layer = model.model.layers[layer_idx].mlp
        
        gate_orig = target_layer.gate_proj.weight.data.clone()
        up_orig   = target_layer.up_proj.weight.data.clone()
        down_orig = target_layer.down_proj.weight.data.clone()

        # 🌟 提取当前层的 S Vector
        s_vec_gate = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.gate_proj")
        s_vec_up   = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.up_proj")
        s_vec_down = activation_scales_dict.get(f"model.layers.{layer_idx}.mlp.down_proj")

        # 🌟 聚类必须使用加权后的特征！
        print("    🔀 计算 Hessian-Aware 联合聚类排列 (CPU)...")
        W_gate_scaled = gate_orig.float() * (s_vec_gate.to(gate_orig.device).float() if s_vec_gate is not None else 1.0)
        W_up_scaled   = up_orig.float()   * (s_vec_up.to(up_orig.device).float()     if s_vec_up is not None else 1.0)
        perm, inv_perm = find_permutation_idx(W_gate_scaled, W_up_scaled)

        # 🧪 Step 2: 在当前层依次跑 4 个模式
        for mode_idx, mode_name in modes:
            with torch.no_grad():
                # 注意：透传 s_vector！
                target_layer.gate_proj.weight.data = compress_matrix(
                    gate_orig, args.ratio, mode_idx, perm, inv_perm, is_down_proj=False,
                    adaptive_mode=args.adaptive_mode, quantum_scale=args.quantum_scale, energy_threshold=args.energy_threshold,
                    s_vector=s_vec_gate
                )
                target_layer.up_proj.weight.data = compress_matrix(
                    up_orig, args.ratio, mode_idx, perm, inv_perm, is_down_proj=False,
                    adaptive_mode=args.adaptive_mode, quantum_scale=args.quantum_scale, energy_threshold=args.energy_threshold,
                    s_vector=s_vec_up
                )
                target_layer.down_proj.weight.data = compress_matrix(
                    down_orig, args.ratio, mode_idx, perm, inv_perm, is_down_proj=True,
                    adaptive_mode=args.adaptive_mode, quantum_scale=args.quantum_scale, energy_threshold=args.energy_threshold,
                    s_vector=s_vec_down
                )
                
            # 跑分并记录
            ppl = eval_ppl(model, tokenizer)
            print(f"    📊 {mode_name:<25} PPL: {ppl}")
            append_to_csv(args.csv_name, layer_idx, mode_name, ppl)
            
            torch.cuda.empty_cache(); gc.collect()

        # 🛡️ Step 3: 满血恢复
        print(f"    🔄 测试完毕，正在将 Layer {layer_idx} 恢复至满血状态...")
        with torch.no_grad():
            target_layer.gate_proj.weight.data.copy_(gate_orig)
            target_layer.up_proj.weight.data.copy_(up_orig)
            target_layer.down_proj.weight.data.copy_(down_orig)
            
        del gate_orig, up_orig, down_orig
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n🎉 所有单层独立消融测试完成！结果已保存在 {args.csv_name}")

if __name__ == "__main__":
    main()