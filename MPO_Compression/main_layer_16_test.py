#!/usr/bin/env python3
"""
MPO 单层消融实验 (Ablation Study)
仅对第 16 层的 MLP 进行 MPO 压缩与局部蒸馏，测试算法在消除误差累积后的真实极限。
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb

# 导入底层组件
from mpo_modules.core import MPOLinear
from test_MPO import factor_linear_mpo_custom, find_factors_balanced
from mpo_modules.factorization import estimate_mpo_bond_dim
from calibration import get_activation_scales

# 之前写好的 ResMPOWrapper 贴在这里 (保持不变，省略展开以节省篇幅)
# class ResMPOWrapper(nn.Module): ... 
# 请把上一版 main_layer_by_layer.py 里的 ResMPOWrapper 完整类代码复制到这里
from main_layer_by_layer import ResMPOWrapper

# ==========================================
# 🔧 配置
# ==========================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
TARGET_LAYER = 16    # 🎯 靶向手术层
TARGET_RATIO = 0.2   # 单层我们可以切狠一点！只保留 20%
LORA_RANK = 32
NUM_CORES = 3
LOCAL_STEPS = 500    # 单层微调 500 步
SAVE_PATH = "./ablation_layer_16_only.pt"

def main():
    print(f"==================================================")
    print(f"🔬 控制变量法：靶向压缩 Layer {TARGET_LAYER}")
    print(f"==================================================")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("\n📦 加载 Teacher 和 Student...")
    teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    student_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    teacher_model.eval()

    print("\n🧬 提取 ASVD 异常激活值...")
    activation_scales_dict = get_activation_scales(student_model, tokenizer, num_samples=64, max_len=512)

    print(f"\n🔪 正在对 Layer {TARGET_LAYER} 进行精确物理切割...")
    student_layer = student_model.model.layers[TARGET_LAYER]
    teacher_layer = teacher_model.model.layers[TARGET_LAYER]

    for proj_name in ["gate_proj", "up_proj"]:
        lin = getattr(student_layer.mlp, proj_name)
        out_f, in_f = lin.weight.shape
        dtype0 = lin.weight.dtype

        # MPO 参数计算
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
        cleaned_cores = [c.to(device=device, dtype=dtype0) for c in cores_list]
        mpo = MPOLinear(in_f, out_f, cleaned_cores, s_vector=s_vector, boundary="open")

        mpo_wrapper = ResMPOWrapper(mpo, in_f, out_f, LORA_RANK, lin.weight)
        setattr(student_layer.mlp, proj_name, mpo_wrapper)
        print(f"    ✅ {proj_name} 切割为 MPO (保留率 {TARGET_RATIO*100}%, Rank {chi_ffn})")

    # 冻结其余 31 层，只训练第 16 层的 MPO 和 LoRA
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

    optimizer = bnb.optim.AdamW8bit(trainable_params, weight_decay=0.01)

    print("\n📚 缓存 Layer 16 的输入数据...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 50).select(range(200))
    cached_inputs = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for text in ds["text"]:
            input_ids = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).input_ids.to(device)
            hidden_states = teacher_model.model.embed_tokens(input_ids)
            for i in range(TARGET_LAYER):
                hidden_states = teacher_model.model.layers[i](hidden_states)[0]
            cached_inputs.append(hidden_states.cpu())

    print(f"\n🔥 启动局部微调 ({LOCAL_STEPS} 步)...")
    student_layer.train()
    progress = tqdm(range(LOCAL_STEPS))
    for step in progress:
        idx = torch.randint(0, len(cached_inputs), (1,)).item()
        h_in = cached_inputs[idx].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                t_out = teacher_layer(h_in)[0]
            s_out = student_layer(h_in)[0]
            loss = torch.nn.functional.mse_loss(s_out, t_out)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            progress.set_postfix({"MSE": f"{loss.item():.4f}"})

    print(f"\n💾 缝合完毕，正在保存...")
    torch.save(student_model.state_dict(), SAVE_PATH)
    print(f"🎉 模型已保存至 {SAVE_PATH}。直接拿去测 PPL 吧！")

if __name__ == "__main__":
    main()