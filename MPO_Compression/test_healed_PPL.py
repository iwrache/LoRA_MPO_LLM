#!/usr/bin/env python3
"""
加载 healing 后的 checkpoint，仅计算 Wikitext-2 PPL，不进行任何训练。
用法示例：
    python test_healed_ppl.py --checkpoint /path/to/checkpoint_upd_1000.pt
"""
import os
os.environ["HF_HOME"] = "/mnt/hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hf-cache/datasets"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import sys

# 确保能导入父目录下的模块
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from mpo_modules.core import MPOLinear

# ==================== ResMPOWrapper 定义 (与主脚本一致) ====================
class ResMPOWrapper(torch.nn.Module):
    def __init__(self, mpo_module, in_features, out_features, lora_rank, W_orig, skip_svd=True, s_vector=None):
        super().__init__()
        self.mpo = mpo_module
        self.r = lora_rank
        target_device, target_dtype = W_orig.device, W_orig.dtype

        # 仅用于加载 checkpoint 的骨架搭建，不需要初始化 LoRA，因为马上会被覆盖
        self.lora_A = torch.nn.Parameter(torch.zeros(lora_rank, in_features, device=target_device, dtype=target_dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, lora_rank, device=target_device, dtype=target_dtype))

    def forward(self, x):
        mpo_out = self.mpo(x)
        lora_out = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return mpo_out + lora_out

# ==================== PPL 计算 (与主脚本完全一致，已修复) ====================
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
    if total_tokens == 0:
        return float("nan")
    return round(torch.exp(torch.tensor(total_loss / total_tokens)).item(), 2)

# ==================== 主逻辑 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="healing 阶段的 checkpoint 路径")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📦 加载基础模型 (bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=None
    ).to("cuda:0")
    base_model.eval()

    print("🧩 根据 checkpoint 重建压缩骨架...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("trainable_state_dict", ckpt)  # 兼容两种格式

    # 遍历每一层，依据 state_dict 的键名动态构建 ResMPOWrapper 或 SVDLinear
    for layer_idx in range(base_model.config.num_hidden_layers):
        student_layer = base_model.model.layers[layer_idx]
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(student_layer.mlp, proj_name)
            in_f, out_f = lin.weight.shape[1], lin.weight.shape[0]

            # 判断是 MPO 还是 SVD
            mpo_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.mpo.cores."
            svd_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.mpo.A.weight"

            if svd_prefix in state_dict:
                # SVD 模式
                from main_svd_baseline import SVDLinear  # 或者把 SVDLinear 也复制到这里
                rank = state_dict[svd_prefix].shape[1]
                mpo = SVDLinear(in_f, out_f, rank, lin.weight, skip_init=True)
            else:
                # MPO 模式
                core_keys = [k for k in state_dict.keys() if k.startswith(mpo_prefix)]
                if len(core_keys) == 0:
                    continue  # 这一层没有被压缩
                cores = []
                for i in range(len(core_keys)):
                    key = f"{mpo_prefix}{i}"
                    cores.append(torch.nn.Parameter(state_dict[key].clone().to(device="cuda:0")))
                mpo = MPOLinear(in_f, out_f, cores)

            # 获取 LoRA rank
            lora_prefix = f"model.layers.{layer_idx}.mlp.{proj_name}.lora_A"
            lora_rank = state_dict[lora_prefix].shape[0] if lora_prefix in state_dict else 32

            # 用 ResMPOWrapper 包裹，skip_svd=True 避免重新初始化
            setattr(student_layer.mlp, proj_name,
                    ResMPOWrapper(mpo, in_f, out_f, lora_rank, lin.weight, skip_svd=True))

    # 加载训练好的权重（包括 LoRA 和投影器等，但投影器在 eval 时不需要）
    print("📥 注入训练好的权重...")
    base_model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict

    # 计算 PPL
    print("🚀 开始评测...")
    base_model.to(torch.bfloat16)
    ppl = eval_ppl(base_model, tokenizer)
    print(f"\n🎯 最终 PPL: {ppl:.2f}")

if __name__ == "__main__":
    main()