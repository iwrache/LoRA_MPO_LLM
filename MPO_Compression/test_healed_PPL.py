#!/usr/bin/env python3
"""
加载压缩骨架与 healing 后的 checkpoint，计算 Wikitext-2 PPL。
用法示例：
    python test_healed_ppl.py \
        --skeleton_ckpt /mnt/sx_data/progressive_layer_checkpoint.pt \
        --healed_ckpt /path/to/healing_checkpoints/checkpoint_upd_1000.pt
"""
import os
os.environ["HF_HOME"] = "/mnt/hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hf-cache/datasets"
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import sys

# 确保能导入 MPO 核心模块
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
from mpo_modules.core import MPOLinear

# ==================== 极简推理版 SVD 骨架 ====================
class SVDLinear(nn.Module):
    """推理专用 SVD 骨架，无需进行耗时的矩阵分解，只做线性前向"""
    def __init__(self, in_features, out_features, rank, W_orig):
        super().__init__()
        target_device, target_dtype = W_orig.device, W_orig.dtype
        self.B = nn.Linear(in_features, rank, bias=False, device=target_device, dtype=target_dtype)
        self.A = nn.Linear(rank, out_features, bias=False, device=target_device, dtype=target_dtype)
        self.bias = None

    def forward(self, x):
        return self.A(self.B(x))

# ==================== ResMPOWrapper 定义 ====================
class ResMPOWrapper(nn.Module):
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig):
        super().__init__()
        self.mpo = mpo_module
        self.r = lora_rank
        target_device, target_dtype = W_orig.device, W_orig.dtype

        # 建立 LoRA 空壳，准备接收加载的权重
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, device=target_device, dtype=target_dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=target_device, dtype=target_dtype))

    def forward(self, x):
        mpo_out = self.mpo(x)
        lora_out = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return mpo_out + lora_out

# ==================== 极其精确的 PPL 计算 ====================
def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=15000):
    model.eval()
    print("⏳ 正在计算 Wikitext-2 PPL...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")

    device = model.model.embed_tokens.weight.device
    input_ids = enc.input_ids.to(device)
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    effective_max = min(max_len, max_ctx)
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    total_loss = 0.0
    total_tokens = 0
    for begin in tqdm(range(0, seq_len, stride), desc="PPL 评测中"):
        end = min(begin + effective_max, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1: continue
        with torch.no_grad():
            out = model(chunk)
            lm_logits = out.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
                
    if total_tokens == 0: return float("nan")
    return round(torch.exp(torch.tensor(total_loss / total_tokens)).item(), 2)

# ==================== 主逻辑 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton_ckpt", type=str, required=True, help="阶段 2 的全模型存档 (包含物理骨架)")
    parser.add_argument("--healed_ckpt", type=str, required=True, help="阶段 4 的 Healing 存档 (仅含满血 LoRA)")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("📦 加载基础模型 (bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=None
    ).to("cuda:0")
    base_model.eval()

    # 🌟 1. 加载包含全部参数的【骨架字典】
    print(f"🧩 正在读取骨架蓝图: {args.skeleton_ckpt}")
    skeleton_ckpt = torch.load(args.skeleton_ckpt, map_location="cpu")
    skeleton_dict = skeleton_ckpt.get("model_state_dict", skeleton_ckpt)

    print("🏗️ 根据骨架蓝图重建物理拓扑...")
    for layer_idx in range(base_model.config.num_hidden_layers):
        student_layer = base_model.model.layers[layer_idx]
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(student_layer.mlp, proj_name)
            in_f, out_f = lin.weight.shape[1], lin.weight.shape[0]

            mpo_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.mpo.cores."
            svd_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.mpo.A.weight"

            # 基于骨架字典来判断模式
            if svd_prefix in skeleton_dict:
                rank = skeleton_dict[svd_prefix].shape[1]
                mpo = SVDLinear(in_f, out_f, rank, lin.weight)
            else:
                core_keys = [k for k in skeleton_dict.keys() if k.startswith(mpo_prefix)]
                if len(core_keys) == 0: continue 
                
                cores = []
                for i in range(len(core_keys)):
                    # 避免硬编码 cuda:0，使用该层原有 device
                    cores.append(nn.Parameter(skeleton_dict[f"{mpo_prefix}{i}"].clone().to(lin.weight.device)))
                mpo = MPOLinear(in_f, out_f, cores)

            lora_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.lora_A"
            lora_rank = skeleton_dict[lora_prefix].shape[0] if lora_prefix in skeleton_dict else 32

            # 用简化的 Wrapper 包裹
            setattr(student_layer.mlp, proj_name, ResMPOWrapper(mpo, in_f, out_f, lora_rank, lin.weight))

    # 🌟 2. 第一重覆盖：注入完整的 MPO/SVD 核心参数
    print("🧱 正在填充底层骨架...")
    base_model.load_state_dict(skeleton_dict, strict=False)
    del skeleton_ckpt, skeleton_dict; gc.collect() if 'gc' in sys.modules else None

    # 🌟 3. 第二重覆盖：注入 Healing 后无敌的 LoRA 参数
    print(f"📥 正在注入 Healed 灵魂 (LoRA): {args.healed_ckpt}")
    healed_ckpt = torch.load(args.healed_ckpt, map_location="cpu")
    healed_dict = healed_ckpt.get("trainable_state_dict", healed_ckpt)
    
    incompatible_keys = base_model.load_state_dict(healed_dict, strict=False)
    print(f"✅ 成功覆盖 {len(healed_dict)} 个满血参数！")
    del healed_ckpt, healed_dict

    # 计算 PPL
    print("\n🚀 骨架与灵魂融合完毕，开始终极评测...")
    base_model.to(torch.bfloat16)
    ppl = eval_ppl(base_model, tokenizer)
    
    print("=" * 50)
    print(f"🎯 最终康复测试 PPL: {ppl:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()