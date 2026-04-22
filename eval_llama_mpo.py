import os
# 强制只在 0, 1, 2 号 GPU 上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import argparse
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.cluster.hierarchy import linkage, leaves_list
from MPO_Compression.calibration import get_activation_scales
# 导入你的 MPO 核心工具
from test_MPO import factor_linear_mpo_custom, reconstruct_mpo_matrix, estimate_bond_dim
import csv
from datetime import datetime
csv_path = "eval_llama_MPO.csv"

def find_permutation_idx(W_gate, W_up):
    """计算 Gate 和 Up 联合特征的最优聚类排列索引 (CPU计算)"""
    features = torch.cat([W_gate, W_up], dim=1).detach().cpu().float().numpy()
    Z = linkage(features, method='ward', metric='euclidean')
    perm = torch.tensor(leaves_list(Z), dtype=torch.long)
    inv_perm = torch.argsort(perm) # 逆排列，用于无损还原
    return perm, inv_perm


def compress_matrix(W_orig, target_ratio, mode, perm=None, inv_perm=None, is_down_proj=False, lora_rank=32, num_cores=3, s_vector=None):
    """
    根据不同模式 (1~4) 压缩矩阵，自带 GPU 光速 SVD 与 CPU 降级保命机制。
    完美支持 Activation-Aware SVD (ASVD) 与重排空间的对齐。
    """
    bond_dim = estimate_bond_dim(W_orig, num_cores, target_ratio)
    
    # [Step A: 物理重排 (仅限模式 3, 4)]
    if mode in [3, 4]:
        if is_down_proj:
            # down_proj 对列重排
            W_perm = W_orig[:, perm].contiguous()
        else:
            # gate/up_proj 对行重排
            W_perm = W_orig[perm, :].contiguous()
    else:
        W_perm = W_orig.contiguous()

    # 🚀 强制转为 float32
    W_compute = W_perm.float()
    
    # ⚡ [关键修复 1]：处理 s_vector 的设备同步与 down_proj 重排对齐
    s_vec_compute = None
    if s_vector is not None:
        raw_s = s_vector.clone().detach().to(device=W_compute.device, dtype=torch.float32)
        
        # ⚡【核心修复：激活截断平滑】
        # 找到当前层激活均值，设定一个合理的下界，防止 1/s 爆炸
        # 过滤掉极小的死神经元噪声，通常取均值的 1% ~ 10%
        s_mean = raw_s.mean()
        lower_bound = s_mean * 0.05  # 5% 的均值作为地板
        
        # 将过于微小的激活值强行托底
        s_vec_compute = torch.clamp(raw_s, min=lower_bound)
        
        # 也可以加一个平滑指数 (类似 AWQ)，让极端值柔和一点
        # s_vec_compute = s_vec_compute ** 0.5 

        # 如果是 down_proj 且发生了重排，因为输入维度(列)变了，s_vector 必须跟着重排！
        if mode in [3, 4] and is_down_proj:
            s_vec_compute = s_vec_compute[perm].contiguous()
        # 如果是 down_proj 且发生了重排，因为输入维度(列)变了，s_vector 必须跟着重排！
        if mode in [3, 4] and is_down_proj:
            s_vec_compute = s_vec_compute[perm].contiguous()

    # 封装 ASVD 逻辑，方便 GPU/CPU 复用
    def do_asvd(Delta, s_vec):
        if s_vec is not None:
            # 使用激活值加权残差的列 (输入特征维度)
            Delta_scaled = Delta * s_safe.unsqueeze(0)
        else:
            Delta_scaled = Delta
            
        U, S, Vh = torch.linalg.svd(Delta_scaled, full_matrices=False)
        return U, S, Vh

    try:
        # ⚡ 尝试 GPU 分解
        cores = factor_linear_mpo_custom(W_compute, bond_dim, num_cores, s_vector=s_vec_compute)
        W_mpo_recon = reconstruct_mpo_matrix(cores)

        if mode in [2, 4]:
            Delta_W = W_compute - W_mpo_recon
            U, S, Vh = do_asvd(Delta_W, s_vec_compute)

    except RuntimeError as e:
        # 🛡️ CPU 降级
        W_cpu = W_compute.cpu()
        s_cpu = s_vec_compute.cpu() if s_vec_compute is not None else None
        
        cores_cpu = factor_linear_mpo_custom(W_cpu, bond_dim, num_cores, s_vector=s_cpu)
        W_mpo_recon_cpu = reconstruct_mpo_matrix(cores_cpu)
        W_mpo_recon = W_mpo_recon_cpu.to(W_orig.device)

        if mode in [2, 4]:
            Delta_W_cpu = W_cpu - W_mpo_recon_cpu
            U_cpu, S_cpu, Vh_cpu = do_asvd(Delta_W_cpu, s_cpu)
            U, S, Vh = U_cpu.to(W_orig.device), S_cpu.to(W_orig.device), Vh_cpu.to(W_orig.device)

    # [Step C: 叠加 ASVD-LoRA 残差]
    if mode in [2, 4]:
        A = U[:, :lora_rank] @ torch.diag(torch.sqrt(S[:lora_rank]))
        B_scaled = torch.diag(torch.sqrt(S[:lora_rank])) @ Vh[:lora_rank, :]
        
        # ⚡ [关键修复 2]：必须把缩放的尺度除回去，还原到真实的权重空间
        if s_vec_compute is not None:
            s_safe = s_vec_compute
            B = B_scaled / s_safe.unsqueeze(0)
        else:
            B = B_scaled
            
        W_approx = W_mpo_recon + (A @ B)
    else:
        W_approx = W_mpo_recon

    # 移回原数据类型
    W_approx = W_approx.to(W_orig.dtype)

    # [Step D: 逆向重排还原]
    if mode in [3, 4]:
        if is_down_proj:
            W_final = W_approx[:, inv_perm].contiguous()
        else:
            W_final = W_approx[inv_perm, :].contiguous()
    else:
        W_final = W_approx

    return W_final


def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=10000):
    """计算 WikiText-2 PPL"""
    print("\n📊 开始跑 WikiText-2 评测...")
    model.eval() 
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    nlls = []
    for begin in tqdm(range(0, seq_len, stride), desc="PPL 评测中", leave=False):
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


def main():
    parser = argparse.ArgumentParser(description="全模型 MPO 压缩测试 (四种模式依次运行)")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--ratio", type=float, default=0.2, help="中间层目标压缩保留率")
    args = parser.parse_args()

    mode_names = {
        1: "纯 MPO 分解",
        2: "MPO + LoRA 残差",
        3: "Permutation (重排) + MPO",
        4: "究极形态: Permutation + MPO + LoRA"
    }

    print("="*70)
    print(f" 🌋 全模型“U型保护”压缩生存测试 (四种模式依次评测)")
    print(f" 🎯 压缩保留率: {args.ratio*100}%")
    print("="*70)

    # 只加载一次 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 存储各模式的 PPL 结果
    results = {}
    
    # 先计算原始满血模型的 PPL 作为基线
    print("\n📊 [Baseline] 加载原始满血模型...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    baseline_ppl = eval_ppl(model_base, tokenizer)
    print(f"🌟 原始满血模型 PPL: {baseline_ppl}")

    print("📊 正在计算激活缩放因子 (ASVD 初始化)...")
    activation_scales = get_activation_scales(model_base, tokenizer, num_samples=32, max_len=256)
    del model_base
    torch.cuda.empty_cache()
    gc.collect()




    # 依次测试四种模式
    for mode in [1,2,3,4]:
        print("\n" + "="*70)
        print(f" 🔧 正在测试模式 [{mode}] : {mode_names[mode]}")
        print("="*70)

        # 重新加载原始模型（保证各模式独立）
        print("📦 加载原始模型...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        layers = model.model.layers
        num_layers = len(layers)
        compressed_count = 0

        print(f"🦴 开始压缩中间层 (共 {num_layers} 层，跳过首尾各3层)...")
        for i in tqdm(range(num_layers), desc=f"Mode {mode} 压缩进度"):
            if i < 3 or i >= (num_layers - 3):
                continue

            target_layer = layers[i].mlp
            with torch.no_grad():
                W_gate = target_layer.gate_proj.weight.data
                W_up   = target_layer.up_proj.weight.data
                W_down = target_layer.down_proj.weight.data

                # 获取该层各投影的激活缩放因子
                s_gate = activation_scales.get(f"model.layers.{i}.mlp.gate_proj")
                s_up   = activation_scales.get(f"model.layers.{i}.mlp.up_proj")
                s_down = activation_scales.get(f"model.layers.{i}.mlp.down_proj")
                
                perm, inv_perm = None, None
                if mode in [3, 4]:
                    perm, inv_perm = find_permutation_idx(W_gate, W_up)

                target_layer.gate_proj.weight.data = compress_matrix(
                    W_gate, args.ratio, mode, perm, inv_perm, is_down_proj=False, s_vector=s_gate)
                target_layer.up_proj.weight.data   = compress_matrix(
                    W_up, args.ratio, mode, perm, inv_perm, is_down_proj=False, s_vector=s_up)
                target_layer.down_proj.weight.data = compress_matrix(
                    W_down, args.ratio, mode, perm, inv_perm, is_down_proj=True, s_vector=s_down)

                compressed_count += 1
                torch.cuda.empty_cache()
                gc.collect()

        print(f"✅ 模式 {mode} 压缩完成，共处理 {compressed_count} 层。")
        ppl = eval_ppl(model, tokenizer)
        results[mode] = ppl
        print(f"📊 模式 {mode} PPL: {ppl}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 准备写入的一行数据
        row = {
            "timestamp": timestamp,
            "mode": mode,
            "mode_name": mode_names[mode],
            "target_ratio": args.ratio,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl
        }

        # 判断文件是否存在，决定是否写入表头
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"📁 结果已保存至 {csv_path}")

    # 打印最终战报
    print("\n" + "="*70)
    print(" 🏆 四种模式零样本 PPL 对比战报")
    print("="*70)
    print(f" 压缩保留率: {args.ratio*100}%")
    print(f" 原始满血模型 PPL: {baseline_ppl}")
    print("-"*70)
    for mode in [1, 2, 3, 4]:
        print(f" [{mode}] {mode_names[mode]:<30}: {results[mode]:>8.2f}")
    print("="*70)
    
if __name__ == "__main__":
    main()