#!/usr/bin/env python3
"""
MPO 单层极限消融实验 (4-GPU 分布式版)
只压缩第 16 层，抛弃 U 型公式，使用固定压缩率。
利用 Forward Hook 完美隔离 MLP，在多卡分布下实现极速局部收敛。
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb

# 💡 强制使用 4 张卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales

# ==========================================
# 🔧 核心配置区
# ==========================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
TARGET_LAYER = 6        # 🎯 只对这一层动刀
TARGET_RATIO = 0.2       # 🎯 固定压缩率 (为了看极限，设为 20%)
LORA_RANK = 32
NUM_CORES = 3
LOCAL_STEPS = 600        # 局部缝合步数
SAVE_PATH = "./ablation_layer16_only_r0.2.pt"


# ==========================================
# 🧩 带奇异值初始化的 ResMPO
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

        print(f"      [Res-MPO] 正在通过 SVD(ΔW) 初始化 LoRA 矩阵 (r={self.r}) on {target_device}...")
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


def main():
    print(f"==================================================")
    print(f"🚀 4-GPU 纯粹消融实验：仅压缩 Layer {TARGET_LAYER}")
    print(f"==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 💡 4卡自动分配！HuggingFace 会把模型平均切到 4 张卡上
    print("\n📦 正在 4 张 GPU 上加载模型 (Teacher & Student)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    student_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    teacher_model.eval()

    # 定位 Layer 16 在哪张卡上
    student_layer = student_model.model.layers[TARGET_LAYER]
    teacher_layer = teacher_model.model.layers[TARGET_LAYER]
    layer_device = student_layer.mlp.gate_proj.weight.device
    print(f"📍 探测完毕：Layer {TARGET_LAYER} 位于物理显卡 {layer_device}")

    print("\n🧬 提取激活向量 (ASVD)...")
    activation_scales_dict = get_activation_scales(student_model, tokenizer, num_samples=32, max_len=256)

    print(f"\n🔪 对 Layer {TARGET_LAYER} 执行固定比例 ({TARGET_RATIO*100}%) MPO 切割...")
    for proj_name in ["gate_proj", "up_proj"]:
        lin = getattr(student_layer.mlp, proj_name)
        out_f, in_f = lin.weight.shape
        dtype0 = lin.weight.dtype

        # 纯粹的固定比例，无 U 型公式
        chi_ffn = estimate_mpo_bond_dim(in_f, out_f, NUM_CORES, TARGET_RATIO)
        out_fac = find_factors_balanced(out_f, NUM_CORES)
        in_fac = find_factors_balanced(in_f, NUM_CORES)

        full_name = f"model.layers.{TARGET_LAYER}.mlp.{proj_name}"
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
        print(f"    ✅ {proj_name} 切割完成 (Rank: {chi_ffn})")

    # 冻结全局，只解锁 Layer 16 的 MPO + LoRA
    for param in student_model.parameters():
        param.requires_grad = False
        
    trainable_params = []
    for proj_name in ["gate_proj", "up_proj"]:
        wrapper = getattr(student_layer.mlp, proj_name)
        wrapper.lora_A.requires_grad = True
        wrapper.lora_B.requires_grad = True
        for core in wrapper.mpo.cores:
            core.requires_grad = True
        trainable_params.append({"params": wrapper.lora_A, "lr": 5e-4})
        trainable_params.append({"params": wrapper.lora_B, "lr": 5e-4})
        trainable_params.append({"params": wrapper.mpo.cores, "lr": 5e-5})

    # 将优化器放在对应的 GPU 上
    optimizer = bnb.optim.AdamW8bit(trainable_params, weight_decay=0.01)

    print("\n📚 截获 Layer 16 的前向输入 (Hook 魔法)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 50).select(range(200))
    
    cached_mlp_inputs = []
    
    # 💡 魔法 Hook：只要数据流进 Teacher 的第 16 层 MLP，我们就把它偷存下来
    def mlp_hook(module, args):
        cached_mlp_inputs.append(args[0].detach().cpu())
        
    handle = teacher_layer.mlp.register_forward_pre_hook(mlp_hook)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for text in tqdm(ds["text"], desc="缓存特征流"):
            input_ids = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).input_ids
            input_ids = input_ids.to(teacher_model.device) # 丢给第一层所在的卡
            teacher_model(input_ids) # 让流水线自己跑，跑到第16层自然会被 Hook 偷走
            
    handle.remove() # 缓存完毕，拆除监听器

    print(f"\n🔥 启动 Layer {TARGET_LAYER} 局部极速微调 ({LOCAL_STEPS} 步)...")
    student_layer.train()
    progress = tqdm(range(LOCAL_STEPS))
    for step in progress:
        idx = torch.randint(0, len(cached_mlp_inputs), (1,)).item()
        # 把数据搬到第 16 层所在的卡上
        h_in = cached_mlp_inputs[idx].to(layer_device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 💡 完美隔离：只过 MLP，不理会 Attention 和 RoPE 掩码！
            with torch.no_grad():
                t_out = teacher_layer.mlp(h_in)
            s_out = student_layer.mlp(h_in)
            loss = F.mse_loss(s_out, t_out)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            progress.set_postfix({"MSE": f"{loss.item():.5f}"})

    print(f"\n💾 单层手术缝合完毕！正在保存...")
    # 保存整个模型（未修改的层是原版，修改的层是缝合后的）
    torch.save(student_model.state_dict(), SAVE_PATH)
    print(f"🎉 存档已保存至: {SAVE_PATH}")
    print("➡️ 下一步：直接用你之前写好的评估脚本，加载这个 pt 测 PPL！")

if __name__ == "__main__":
    main()