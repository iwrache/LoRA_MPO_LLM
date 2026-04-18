#!/usr/bin/env python3
"""
MPO 多层自定义列表压缩与逐层缝合管线
用法示例:
python main_custom_multi_layers.py \
    --layers 16 18 20 \
    --ratios 0.2 0.3 0.15 \
    --local_steps 500 \
    --save_path ./custom_layers_healed.pt
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb

# 强制使用 4 张卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales

# ==========================================
# 🧩 带奇异值初始化的 ResMPOWrapper
# ==========================================
class ResMPOWrapper(nn.Module):
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig, skip_svd=False):
        super().__init__()
        self.mpo = mpo_module
        self.r = lora_rank
        target_device = W_orig.device
        target_dtype = W_orig.dtype
        
        if skip_svd:
            self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, device=target_device, dtype=target_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=target_device, dtype=target_dtype))
            return

        with torch.no_grad():
            mpo_cpu = self.mpo.cpu().float()
            if hasattr(mpo_cpu, 's_vector') and mpo_cpu.s_vector is not None:
                mpo_cpu.s_vector = mpo_cpu.s_vector.cpu().float()
            bias_backup = None
            if hasattr(mpo_cpu, 'bias') and mpo_cpu.bias is not None:
                bias_backup = mpo_cpu.bias.data.clone()
                mpo_cpu.bias.data.zero_()

            eye = torch.eye(in_features, dtype=torch.float32)  
            W_mpo = mpo_cpu(eye).T                             
            if bias_backup is not None:
                mpo_cpu.bias.data.copy_(bias_backup)

            Delta_W = W_orig.cpu().float() - W_mpo
            U, S, Vh = torch.linalg.svd(Delta_W, full_matrices=False)
            S_sqrt = torch.diag(torch.sqrt(S[:self.r].clamp(min=0)))
            B_init = U[:, :self.r] @ S_sqrt
            A_init = S_sqrt @ Vh[:self.r, :]
            del eye, W_mpo, Delta_W, U, S, Vh

        self.lora_A = nn.Parameter(A_init.to(target_dtype).to(target_device))
        self.lora_B = nn.Parameter(B_init.to(target_dtype).to(target_device))
        self.mpo = self.mpo.to(device=target_device, dtype=target_dtype)
        if hasattr(self.mpo, 's_vector') and self.mpo.s_vector is not None:
            self.mpo.s_vector = self.mpo.s_vector.to(device=target_device, dtype=target_dtype)

    def forward(self, x):
        return self.mpo(x) + F.linear(F.linear(x, self.lora_A), self.lora_B)

# ==========================================
# 🚀 核心管线
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # 使用 nargs='+' 接收列表输入
    parser.add_argument("--layers", type=int, nargs='+', required=True, help="要压缩的层列表 (例如: 16 18 20)")
    parser.add_argument("--ratios", type=float, nargs='+', required=True, help="对应的保留率列表 (例如: 0.2 0.3 0.15)")
    parser.add_argument("--local_steps", type=int, default=500, help="每层的局部缝合步数")
    parser.add_argument("--save_path", type=str, required=True, help="保存路径")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()

    if len(args.layers) != len(args.ratios):
        raise ValueError("❌ layers 和 ratios 的列表长度必须完全一致！")

    print(f"==================================================")
    print(f"🚀 多层自定义列表手术台启动")
    print(f"   目标层级: {args.layers}")
    print(f"   对应保留率: {args.ratios}")
    print(f"==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("\n📦 正在 4 张 GPU 上加载模型 (Device Map Auto)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    student_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    teacher_model.eval()

    print("\n🧬 提取全局激活向量 (ASVD)...")
    activation_scales_dict = get_activation_scales(student_model, tokenizer, num_samples=32, max_len=256)

    # 准备校准数据集
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 50).select(range(200))
    encoded_texts = [tokenizer(text, return_tensors="pt", max_length=256, truncation=True).input_ids for text in ds["text"]]

    # 循环遍历用户提供的层和保留率列表
    for i, (layer_idx, ratio) in enumerate(zip(args.layers, args.ratios)):
        print(f"\n" + "="*50)
        print(f" 🎯 [任务 {i+1}/{len(args.layers)}] 正在处理 Layer {layer_idx} (保留率 {ratio*100}%)")
        print("="*50)

        student_layer = student_model.model.layers[layer_idx]
        teacher_layer = teacher_model.model.layers[layer_idx]
        layer_device = student_layer.mlp.gate_proj.weight.device

        # 1. 物理切割
        NUM_CORES, LORA_RANK = 3, 32
        for proj_name in ["gate_proj", "up_proj"]:
            lin = getattr(student_layer.mlp, proj_name)
            out_f, in_f = lin.weight.shape
            dtype0 = lin.weight.dtype

            chi_ffn = estimate_mpo_bond_dim(in_f, out_f, NUM_CORES, ratio)
            out_fac, in_fac = find_factors_balanced(out_f, NUM_CORES), find_factors_balanced(in_f, NUM_CORES)
            full_name = f"model.layers.{layer_idx}.mlp.{proj_name}"
            s_vector = activation_scales_dict.get(full_name) if activation_scales_dict else None
            
            W = lin.weight.detach().clone().cpu().float()
            s_vector_cpu = s_vector.cpu().float() if s_vector is not None else None

            cores_list = factor_linear_mpo_custom(
                weight=W, bond_dim=chi_ffn, num_cores=NUM_CORES,
                out_fac=out_fac, in_fac=in_fac,
                s_vector=s_vector_cpu, boundary="open", adaptive=True, energy_threshold=0.99
            )
            cleaned_cores = [c.to(device=layer_device, dtype=dtype0) for c in cores_list]
            mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vector, boundary="open")

            mpo_wrapper = ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight)
            setattr(student_layer.mlp, proj_name, mpo_wrapper)
            print(f"    ✅ {proj_name} 切割完成 (Bond Dim: {chi_ffn})")

        # 2. 配置优化器 (冻结全局，只解冻当前层的 MPO)
        for param in student_model.parameters(): param.requires_grad = False
        trainable_params = []
        for proj_name in ["gate_proj", "up_proj"]:
            wrapper = getattr(student_layer.mlp, proj_name)
            wrapper.lora_A.requires_grad = True
            wrapper.lora_B.requires_grad = True
            for core in wrapper.mpo.cores: core.requires_grad = True
            trainable_params.append({"params": wrapper.lora_A, "lr": 5e-4})
            trainable_params.append({"params": wrapper.lora_B, "lr": 5e-4})
            trainable_params.append({"params": wrapper.mpo.cores, "lr": 5e-5})
        optimizer = bnb.optim.AdamW8bit(trainable_params, weight_decay=0.01)

        # 3. 截胡缓存 (隔离误差，用 Teacher 的输入喂给当前层)
        print("    📚 正在截获本层前向输入 (Hook)...")
        cached_mlp_inputs = []
        def mlp_hook(module, hook_args):
            cached_mlp_inputs.append(hook_args[0].detach().cpu())
            
        handle = teacher_layer.mlp.register_forward_pre_hook(mlp_hook)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for input_ids in tqdm(encoded_texts, desc="运行流水线", leave=False):
                input_ids = input_ids.to(teacher_model.device)
                teacher_model(input_ids)
        handle.remove()

        # 4. 局部极速缝合
        print(f"    🔥 启动局部缝合训练 ({args.local_steps} 步)...")
        student_layer.train()
        progress = tqdm(range(args.local_steps), desc=f"Layer {layer_idx} Healing")
        for step in progress:
            idx = torch.randint(0, len(cached_mlp_inputs), (1,)).item()
            h_in = cached_mlp_inputs[idx].to(layer_device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad(): t_out = teacher_layer.mlp(h_in)
                s_out = student_layer.mlp(h_in)
                loss = F.mse_loss(s_out, t_out)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0: progress.set_postfix({"MSE": f"{loss.item():.5f}"})

        # 清理内存，准备进入下一层循环
        del cached_mlp_inputs
        torch.cuda.empty_cache()
        print(f"    ✅ Layer {layer_idx} 缝合完毕！")

    print(f"\n🎉 所有指定层均已处理完毕！")
    print(f"💾 正在保存最终模型至: {args.save_path}")
    torch.save(student_model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()