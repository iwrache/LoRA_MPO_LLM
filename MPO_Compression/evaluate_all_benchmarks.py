#!/usr/bin/env python3
"""
evaluate_all_benchmarks.py

评测 LLaMA-7B: PPL (Wiki/PTB/C4) + Zero-Shot (7个任务)

用法:
  pip install lm-eval datasets

  # 评测原始模型
  python evaluate_all_benchmarks.py --model_path /path/to/llama-7b --mode original

  # 评测压缩模型 (需要你自己补充加载代码)
  python evaluate_all_benchmarks.py --model_path /path/to/llama-7b --mode compressed --checkpoint xxx.pt
"""
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import os
import json
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import logging
logging.getLogger("lm-eval").setLevel(logging.ERROR)


# =============================================================
#  Part 1: Perplexity 评测 (Wiki / PTB / C4)
# =============================================================

@torch.no_grad()
def evaluate_ppl(model, tokenizer, dataset_name, max_seq_len=2048, stride=512):
    from datasets import load_dataset

    model.eval()

    # ---------- 确定输入设备 ----------
    device = model.model.embed_tokens.weight.device
    
    print(f"\n{'='*55}")
    print(f"  📐 评测 {dataset_name.upper()} Perplexity")
    print(f"{'='*55}")

    # ---------- 加载数据 ----------
    t0 = time.time()
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        text = "\n\n".join([t for t in ds["text"] if t.strip()])

    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test", revision="refs/convert/parquet")
        text = " ".join(ds["sentence"])

    elif dataset_name == "c4":
        print("  ⏳ C4 streaming 加载中 (取前 256 条)...")
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= 256:
                break
            texts.append(ex["text"])
        text = "\n\n".join(texts)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # ---------- 编码 ----------
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids          # [1, total_tokens]
    total_tokens = input_ids.size(1)
    print(f"  总 token 数: {total_tokens:,}")

    # ---------- 滑动窗口计算 NLL ----------
    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, total_tokens, stride), desc=f"  {dataset_name}"):
        end = min(begin + max_seq_len, total_tokens)
        trg_len = end - prev_end

        ids    = input_ids[:, begin:end].to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100          # 只对新窗口部分算 loss

        outputs = model(ids, labels=labels)
        nlls.append((outputs.loss.float() * trg_len).item()) # 👈 加上 .item()

        prev_end = end
        if end == total_tokens:
            break

    ppl = torch.exp(torch.tensor(sum(nlls)) / prev_end).item()
    elapsed = time.time() - t0
    print(f"  ✅ {dataset_name.upper()} PPL = {ppl:.2f}  ({elapsed:.0f}s)")
    return round(ppl, 2)


# =============================================================
#  Part 2: Zero-Shot Accuracy (7 个任务, 使用 lm-eval)
# =============================================================

def evaluate_zero_shot(model, tokenizer, batch_size=8):
    """
    需要: pip install lm-eval>=0.4.0
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("\n❌ pip install lm-eval>=0.4.0")
        return None

    TASKS = [
        "openbookqa",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "hellaswag",
        "piqa",
        #"mathqa",
    ]

    # 任务名 → 打印用的短名
    SHORT = {
        "openbookqa":    "Openb.",
        "arc_easy":      "ARC_e",
        "arc_challenge": "ARC_c",
        "winogrande":    "WinoG.",
        "hellaswag":     "HellaS.",
        "piqa":          "PIQA",
        #"mathqa":        "MathQA",
    }

    # 每个任务应该用 acc 还是 acc_norm
    USE_NORM = {
        "openbookqa":    True,
        "arc_easy":      True,
        "arc_challenge": True,
        "winogrande":    False,   # winogrande 不做 length-norm
        "hellaswag":     True,
        "piqa":          True,
        #"mathqa":        True,
    }

    print(f"\n{'='*55}")
    print(f"  🎯 Zero-Shot 评测 ({len(TASKS)} 个任务)")
    print(f"{'='*55}")

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=TASKS,
        num_fewshot=0,
        batch_size=batch_size,
    )

    accuracies = {}
    for task in TASKS:
        res = results["results"].get(task, {})

        if USE_NORM[task]:
            acc = (res.get("acc_norm,none")
                   or res.get("acc_norm")
                   or res.get("acc,none")
                   or res.get("acc")
                   or 0)
        else:
            acc = (res.get("acc,none")
                   or res.get("acc")
                   or 0)

        accuracies[task] = round(acc * 100, 1)
        print(f"  ✅ {SHORT[task]:>8s}: {accuracies[task]:.1f}%")

    avg = round(np.mean(list(accuracies.values())), 1)
    accuracies["average"] = avg
    print(f"  {'Avg.':>10s}: {avg:.1f}%")
    return accuracies


# =============================================================
#  Part 3: 格式化输出 (对标论文 Table 1)
# =============================================================

def print_table(rows):
    """
    rows: list of dict, 每个 dict 包含:
      name, ppl: {wikitext, ptb, c4}, zs: {openbookqa, ..., average}
    """
    TASK_ORDER = ["openbookqa","arc_easy","arc_challenge",
                  "winogrande","hellaswag","piqa","mathqa"]
    SHORT      = ["Openb.","ARC_e","ARC_c","WinoG.","HellaS.","PIQA","MathQA"]

    print("\n" + "="*120)
    print(" 📊  Table 1 对比  (Perplexity ↓ | Zero-Shot Accuracy % ↑)")
    print("="*120)

    hdr = f"{'Method':<22}|{'Wiki':>7}{'PTB':>8}{'C4':>8} |"
    for s in SHORT:
        hdr += f"{s:>8}"
    hdr += f" |{'Avg.':>7}{'Drop':>7}"
    print(hdr)
    print("-"*120)

    # 取第一行作为 baseline 算 Drop
    baseline_avg = rows[0].get("zs", {}).get("average", None)

    for r in rows:
        ppl = r.get("ppl", {})
        zs  = r.get("zs",  {})

        def f_ppl(v): return f"{v:>7.2f}" if isinstance(v, (int,float)) else f"{'—':>7}"
        def f_acc(v): return f"{v:>8.1f}" if isinstance(v, (int,float)) else f"{'—':>8}"

        line = f"{r['name']:<22}|{f_ppl(ppl.get('wikitext'))}{f_ppl(ppl.get('ptb'))}{f_ppl(ppl.get('c4'))} |"
        for t in TASK_ORDER:
            line += f_acc(zs.get(t))

        avg = zs.get("average")
        line += f" |{f_acc(avg)}"

        if baseline_avg and avg and baseline_avg > 0:
            drop = (1 - avg / baseline_avg) * 100
            line += f"{drop:>6.1f}%"
        else:
            line += f"{'—':>7}"

        print(line)

    print("="*120)


# =============================================================
#  Part 4: 模型加载
# =============================================================

def load_original(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n📦 加载原始 LLaMA-7B: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def load_compressed(model_path, checkpoint_path):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from healing import load_checkpoint
    from main import compress_with_function2

    print(f"\n📦 1. 正在加载满血原版模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # 既然我们要 skip_svd，模型放 CPU 就行了，防爆显存
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        device_map="cpu", 
    )

    print(f"📂 2. 提前读取 Checkpoint 以获取真实截断维度...")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise ValueError(f"❌ 找不到 Checkpoint 文件: {checkpoint_path}")
    
    # 💡 提前加载全家桶
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    trainable_weights = ckpt.get("trainable_state_dict", ckpt)

    print("✂️ 3. 正在执行 MPO 空壳搭建 (根据真实权重动态定型)...")
    cfg = {
        "num_cores": 3,
        "boundary": "open",
        "freeze_blocks": 4,  
        "mid_blocks": 16,
        "target_ratio": 0.6,
        "deep_ratio": 0.3,  
        "skip_mlp": "down_proj",
        "lora_rank": 32,
        "head_ratio": 0.5 
    }
    
    # 💡 终极绝招：开启 skip_svd，并且把 trainable_weights 传进去给它“抄作业”！
    model = compress_with_function2(
        model, 
        cfg, 
        activation_scales=None, 
        skip_svd=True, 
        loaded_weights=trainable_weights
    )

    print(f"📥 4. 正在将权重注入量身定制的 MPO 骨架...")
    # 这里你的 load_checkpoint 里会执行 model.load_state_dict(strict=False)
    # 因为刚才骨架是看着权重搭建的，这次绝对是 100% 完美咬合！
    load_checkpoint(model, checkpoint_path)

    print("🚀 5. 满血复活！推入显卡...")
    model = model.cuda()
    model.eval()
    
    return model, tokenizer


# =============================================================
#  Part 5: Main
# =============================================================

def run_full_eval(model, tokenizer, name, max_seq_len=2048, batch_size=8,
                  skip_ppl=False, skip_zs=False):
    """对一个模型跑全部评测，返回结果 dict"""
    result = {"name": name, "ppl": {}, "zs": {}}

    if not skip_ppl:
        for ds in ["wikitext", "ptb", "c4"]:
            result["ppl"][ds] = evaluate_ppl(
                model, tokenizer, ds,
                max_seq_len=max_seq_len, stride=512
            )

    if not skip_zs:
        zs = evaluate_zero_shot(model, tokenizer, batch_size=batch_size)
        if zs:
            result["zs"] = zs

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", choices=["original", "compressed", "both"], default="original")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_zs", action="store_true")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    all_results = []

    # ---- 原始模型 ----
    if args.mode in ["original", "both"]:
        model, tokenizer = load_original(args.model_path)
        r = run_full_eval(model, tokenizer, "Original LLaMA-7B",
                          args.max_seq_len, args.batch_size,
                          args.skip_ppl, args.skip_zs)
        all_results.append(r)

        if args.mode == "both":
            del model; torch.cuda.empty_cache()

    # ---- 压缩模型 ----
    if args.mode in ["compressed", "both"]:
        model, tokenizer = load_compressed(args.model_path, args.checkpoint)
        r = run_full_eval(model, tokenizer, "MPO (Ours)",
                          args.max_seq_len, args.batch_size,
                          args.skip_ppl, args.skip_zs)
        all_results.append(r)

    # ---- 输出 ----
    print_table(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存至 {args.output}")