# healing.py
import os

# 🚨 必须在 import 之前设置环境变量
os.environ["HF_HOME"] = "/mnt/hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hf-cache/datasets"

from datasets import load_dataset, concatenate_datasets

# 增加 cache_dir 双保险
CACHE_DIR = "/mnt/hf-cache/datasets"
os.makedirs(CACHE_DIR, exist_ok=True)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import bitsandbytes as bnb
import random
from accelerate import Accelerator


def quick_evaluate_generation(student_model, teacher_model, tokenizer, prompts, accelerator=None):
    """极速对比评估：针对 DDP 多卡环境做了防刷屏和解包优化"""
    
    # 🌟 防刷屏护盾：只允许主显卡 (GPU 0) 打印和生成，其他显卡直接跳过
    if accelerator is not None and not accelerator.is_main_process:
        return

    print(f"\n{'='*60}\n👀 [双轨质量评估] 正在生成 Teacher vs Student 对比文本...\n{'-'*60}")
    
    # 🌟 解包护盾：DDP 包装的模型不能直接调用 .generate()，必须先解包！
    s_model = accelerator.unwrap_model(student_model) if accelerator else student_model
    t_model = accelerator.unwrap_model(teacher_model) if (accelerator and teacher_model) else teacher_model
    
    s_model.eval()
    if t_model is not None:
        t_model.eval()
    
    device = next(s_model.parameters()).device
    
    for prompt in prompts:
        chat_prompt = f"Question: {prompt}\nAnswer:"
        print(f"👤 Question: {prompt}\n")
        
        # 1. 跑原版 Teacher
        if t_model is not None:
            inputs_t = tokenizer(chat_prompt, return_tensors="pt").to(device)
            # 保持和训练一致的 bf16 精度
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                t_outputs = t_model.generate(
                    **inputs_t, max_new_tokens=60, do_sample=False,
                    repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id
                )
            t_text = tokenizer.decode(t_outputs[0][inputs_t.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"👑 [原版 Teacher]: {t_text.strip()}\n")
        
        # 2. 跑压缩 Student
        inputs_s = tokenizer(chat_prompt, return_tensors="pt").to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_outputs = s_model.generate(
                **inputs_s, max_new_tokens=60, do_sample=False,
                repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id
            )
        s_text = tokenizer.decode(s_outputs[0][inputs_s.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖 [压缩 Student]: {s_text.strip()}\n")
        print("-" * 60)
    
    s_model.train() # 评估完切回训练模式
    print("="*60 + "\n")
import glob
import os

def save_checkpoint(model, feat_projectors, optimizer, scheduler, epoch, step, global_update_step, checkpoint_dir, accelerator=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    actual_model = accelerator.unwrap_model(model) if accelerator else model
    actual_proj = accelerator.unwrap_model(feat_projectors) if accelerator else feat_projectors # 解包投影器
    
    trainable_state_dict = {
        name: param.data.cpu()
        for name, param in actual_model.named_parameters() 
        if param.requires_grad
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_upd_{global_update_step}.pt")
    print(f"💾 保存轻量级 Checkpoint 至 {checkpoint_path}...")
    torch.save({
        'epoch': epoch,
        'step': step,
        'global_update_step': global_update_step,
        'trainable_state_dict': trainable_state_dict,
        'proj_state_dict': actual_proj.state_dict() if actual_proj else None, # 👈 新增：保存翻译官的记忆
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, checkpoint_path)
    
    # 滚动删除...
    existing_ckpts = glob.glob(os.path.join(checkpoint_dir, "checkpoint_upd_*.pt"))
    existing_ckpts.sort(key=os.path.getmtime)
    while len(existing_ckpts) > 1:
        oldest_ckpt = existing_ckpts.pop(0)
        os.remove(oldest_ckpt)
        print(f"🗑️ 已清理过期存档: {oldest_ckpt}")


def load_checkpoint(model, feat_projectors, checkpoint_path, optimizer=None, scheduler=None, accelerator=None, load_optimizer=True): 
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 找不到 checkpoint: {checkpoint_path}")
        return None
    
    print(f"📂 加载 Checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 🌟 核心修复：同理，加载前必须解包目标模型，才能保证名字完美对齐
    actual_model = accelerator.unwrap_model(model) if accelerator else model
    actual_proj = accelerator.unwrap_model(feat_projectors) if accelerator else feat_projectors
    
    if 'trainable_state_dict' in ckpt:
        trainable_weights = ckpt['trainable_state_dict']
        incompatible_keys = actual_model.load_state_dict(trainable_weights, strict=False)
        print(f"✅ 在文件里找到了 {len(trainable_weights)} 个可训练参数。")
        
        num_rejected = len(incompatible_keys.unexpected_keys)
        if num_rejected > 0:
            print(f"🚨 警告：有 {num_rejected} 个参数被拒收了！ 👉 {incompatible_keys.unexpected_keys[:3]}")
    else:
        actual_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # 👈 新增：如果存档里有翻译官的记忆，就唤醒它
    if actual_proj and 'proj_state_dict' in ckpt and ckpt['proj_state_dict']:
        actual_proj.load_state_dict(ckpt['proj_state_dict'])
        print("✅ 成功恢复特征投影器 (Projectors) 的记忆！")

    if load_optimizer:
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    # 总是返回步数信息
    return {'epoch': ckpt.get('epoch', 0), 'global_update_step': ckpt.get('global_update_step', 0)}


# ==========================================
# 🌟 升级 1：支持纯预训练数据(无Prompt)和SFT数据混合的 Dataset
# ==========================================
class MixedHealingDataset(Dataset):
    def __init__(self, tokenizer, max_samples=50000, max_len=2048, custom_dataset=None):
        self.data = []
        count = 0
        
        for item in custom_dataset:
            if count >= max_samples: 
                break
            messages = item["messages"]
            
            # 判断是 PT (纯文本) 还是 SFT (对话)
            if len(messages) == 1 and messages[0]["role"] == "text":
                # 纯预训练数据：全量计算 Loss
                text = messages[0]["content"]
                input_ids = tokenizer(text, max_length=max_len, truncation=True, add_special_tokens=True).input_ids
                labels = list(input_ids)
            else:
                # SFT 对话数据：Mask 掉 User 的部分
                msgs = [m for m in messages if m["role"] != "system"]
                if len(msgs) < 2: continue
                
                prompt_str = tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
                
                ans_str = msgs[1]["content"] + tokenizer.eos_token
                ans_ids = tokenizer(ans_str, add_special_tokens=False).input_ids
                
                input_ids = prompt_ids + ans_ids
                labels = [-100] * len(prompt_ids) + ans_ids
                
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                
            if not any(l != -100 for l in labels):
                continue 
            
            self.data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long), 
                "labels": torch.tensor(labels, dtype=torch.long)
            })
            count += 1
                
        print(f"✅ 成功构建 {len(self.data)} 条黄金康复数据 (序列长度: {max_len})！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def chat_collate_fn(batch, pad_token_id):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = torch.tensor([len(ids) for ids in input_ids])
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    max_len = input_ids_padded.size(1)
    attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return input_ids_padded, labels_padded, attention_mask.long()

# ==========================================
# 🌟 升级 2：引入预训练生语料 (5万条终极配方)
# ==========================================
def get_mixed_healing_dataset(seed=42):
    print("🧪 正在调配大模型终极康复药剂 (5万条 PT+SFT 混合数据集)...")

    # 🌟 1. 骨骼重塑：使用纯本地的高质量 FineWeb-Edu 数据
    print("   -> 抽取 25,000 条顶级教育预训练数据 (本地 FineWeb-Edu)...")
    
    local_fineweb_path = "/mnt/hf-cache/datasets/fineweb_local/fineweb-edu-00000.parquet"
    
    # 使用 parquet 引擎加载
    pt_ds = load_dataset("parquet", data_files=local_fineweb_path, split="train[:25000]")
    
    # 将它的 'text' 字段转化为标准的 messages 格式
    def format_pt(example): 
        return {"messages": [{"role": "text", "content": example["text"]}]}
        
    # 丢弃掉 fineweb 原本自带的各种杂乱字段 (如 url, id, token_count 等)
    pt_ds = pt_ds.map(format_pt, remove_columns=pt_ds.column_names)

    # 2. 情商恢复：UltraChat (30% - 15,000条)
    print("   -> 抽取 15,000 条纯对话数据 (UltraChat)...")
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:15000]")
    chat_ds = chat_ds.select_columns(["messages"])

    # 3. 理科大脑：GSM8K (10% - 5,000条)
    print("   -> 抽取 5,000 条数学逻辑数据 (GSM8K)...")
    math_ds = load_dataset("gsm8k", "main", split="train[:5000]")
    def format_math(example):
        return {"messages": [{"role": "user", "content": "Solve this:\n" + example["question"]},
                             {"role": "assistant", "content": example["answer"]}]}
    math_ds = math_ds.map(format_math, remove_columns=math_ds.column_names)

    # 4. 代码结构：CodeAlpaca (10% - 5,000条)
    print("   -> 抽取 5,000 条代码语法数据 (CodeAlpaca)...")
    code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:5000]")
    def format_code(example):
        prompt = example["instruction"] + ("\n" + example["input"] if example["input"] else "")
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": example["output"]}]}
    code_ds = code_ds.map(format_code, remove_columns=code_ds.column_names)

    print("🔄 放入炼丹炉强力混合 (Shuffle)...")
    mixed_ds = concatenate_datasets([pt_ds, chat_ds, math_ds, code_ds]).shuffle(seed=seed)
    return mixed_ds

# ==========================================
# 🌟 升级 3：引入 Accelerator 满血 3 卡并行与退火蒸馏
# ==========================================
# 🌟 加上 accelerator 参数
def train_healing(student_model, tokenizer, teacher_model=None, dataset_name="mixed", 
                  epochs=1, batch_size=2, accum_steps=8, lr=1e-4, seq_len=2048,
                  save_every_n_steps=250, resume_from_checkpoint=None, 
                  checkpoint_dir="./healing_checkpoints", max_update_steps=1200,
                  accelerator=None, tune_mpo=False): 
    # 🌟 开启梯度检查点 (核武器级显存优化)
    student_model.gradient_checkpointing_enable()
    # ⚠️ 开启检查点时，必须关闭 KV Cache
    student_model.config.use_cache = False
    
    # 🚀 如果外部传进来了 accelerator 就用外部的，没传就自己建一个（方便单卡调试）
    if accelerator is None:
        accelerator = Accelerator(gradient_accumulation_steps=accum_steps, mixed_precision="bf16")
    
    device = accelerator.device

    accelerator.print("\n" + "="*70)
    accelerator.print(f"🚀 [多卡满载] 端到端微调启动! 当前使用 GPU 数量: {accelerator.num_processes}")
    accelerator.print("="*70)
    
    for param in student_model.parameters(): param.requires_grad = False
        
    trainable_params_count = 0
    lora_params, mpo_params = [], []
    
    for name, param in student_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                lora_params.append(param)
                trainable_params_count += param.numel()
            elif "core" in name:
                if tune_mpo:
                    # 🌟 坚决冻结 MPO 骨架！防止连乘导致的梯度核爆！
                    param.requires_grad = True
                    mpo_params.append(param)
                    trainable_params_count += param.numel()
                else:
                    param.requires_grad = False

            
    accelerator.print(f"🔓 解冻参数量: {trainable_params_count/1e6:.2f}M")

    teacher_model.eval()
    for param in teacher_model.parameters(): param.requires_grad = False

    if hasattr(student_model, "enable_input_require_grads"):
        student_model.enable_input_require_grads()

    raw_dataset = get_mixed_healing_dataset()
    dataset = MixedHealingDataset(tokenizer, max_len=seq_len, custom_dataset=raw_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda b: chat_collate_fn(b, tokenizer.pad_token_id))
    
    num_layers = len(student_model.model.layers)
    feat_layers = [i for i in range(0, num_layers, max(1, num_layers // 4))]
    if (num_layers - 1) not in feat_layers: feat_layers.append(num_layers - 1)

    hidden_size = student_model.config.hidden_size

    # 🌟 升级 Projector 为 MLP 缓冲层，解决特征维度扭曲瓶颈
    feat_projectors = nn.ModuleList([
        nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False)
        ) for _ in feat_layers
    ])

    optimizer_grouped_parameters = [
        {"params": lora_params, "lr": lr},
        {"params": feat_projectors.parameters(), "lr": lr}
    ]
    if tune_mpo and mpo_params:
        optimizer_grouped_parameters.append({"params": mpo_params, "lr": lr * 0.1})  # MPO 学习率更低
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, weight_decay=0.01)
    
    effective_total = min((len(dataloader) // accum_steps) * epochs, max_update_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=effective_total)
    
    # 🚀 将所有组件丢给 Accelerator 施加多卡魔法
    student_model, feat_projectors, optimizer, dataloader, scheduler = accelerator.prepare(
        student_model, feat_projectors, optimizer, dataloader, scheduler
    )
    teacher_model = accelerator.prepare(teacher_model)

    student_model.train()
    t0 = time.time()
    global_update_step = 0

    # 🌟 核心修复 1：必须在 prepare 之后，重新收集属于 DDP 模型的参数指针！
    trainable_params = [p for p in student_model.parameters() if p.requires_grad] + list(feat_projectors.parameters())
    
    if resume_from_checkpoint is not None:
        # 仅加载权重，不恢复任何训练状态
        resume_info = load_checkpoint(student_model, feat_projectors, resume_from_checkpoint,
                                    optimizer, scheduler, accelerator, load_optimizer=False)
        # 强制重置 global_update_step（即从头开始计数）
        global_update_step = 0
            
    for epoch in range(epochs):
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            input_ids, labels, attention_mask = batch[0], batch[1], batch[2]

            # 🌟 动态 Loss 权重 (退火蒸馏)
            progress = global_update_step / max_update_steps
            if progress < 0.3:
                # 前 30% 阶段：先修骨骼 (狂拉 Feature)
                lam_feat, lam_log, lam_task = 1.0, 0.1, 0.0
            elif progress < 0.7:
                # 中期：模仿神态 (狂拉 Logits)
                lam_feat, lam_log, lam_task = 0.1, 1.0, 0.1
            else:
                # 后期：独立行走 (回归真实 CE 任务标签)
                lam_feat, lam_log, lam_task = 0.0, 0.5, 1.0

            with accelerator.accumulate(student_model):
                with torch.no_grad():
                    # 🌟 安全护盾：确保输入数据传到了 Teacher 所在的卡上
                    teacher_dev = next(teacher_model.parameters()).device
                    t_outputs = teacher_model(
                        input_ids.to(teacher_dev), 
                        attention_mask=attention_mask.to(teacher_dev), 
                        output_hidden_states=True
                    )
                
                s_outputs = student_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                loss_task = s_outputs.loss if s_outputs.loss is not None else torch.tensor(0.0, device=device)
                
                # --- Top-K Logit 蒸馏 ---
                T = 2.0
                s_logits = s_outputs.logits / T
                with torch.no_grad():
                    t_logits = t_outputs.logits / T
                    K = 128
                    t_topk_vals, t_topk_indices = t_logits.topk(K, dim=-1)
                    mask_t_logits = torch.full_like(t_logits, float('-inf'))
                    mask_t_logits.scatter_(dim=-1, index=t_topk_indices, src=t_topk_vals)
                    t_probs = F.softmax(mask_t_logits, dim=-1)

                s_log_probs = F.log_softmax(s_logits, dim=-1)
                kl_loss = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1) 
                
                num_tokens = attention_mask.sum().clamp(min=1)
                
                with torch.no_grad():
                    mean_kl = (kl_loss * attention_mask).sum() / num_tokens
                    focal_weight = (kl_loss / (mean_kl + 1e-6)).clamp(min=1.0, max=3.0)
                
                valid_kl = kl_loss * focal_weight * attention_mask
                loss_logit = (valid_kl.sum() / num_tokens) * (T ** 2)
                
                # --- MLP 缓冲层 Feature 蒸馏 ---
                loss_feat = torch.tensor(0.0, device=device)
                
                # 🌟 核心修复 1：智能剥离 DDP 外壳，完美兼容多卡环境
                actual_projectors = feat_projectors.module if hasattr(feat_projectors, "module") else feat_projectors
                
                for i, layer_idx in enumerate(feat_layers):
                    # 获取该层的输出，并统一拉到当前卡
                    s_h = s_outputs.hidden_states[layer_idx + 1].to(device)
                    t_h = t_outputs.hidden_states[layer_idx + 1].to(device)
                    
                    if s_h.shape[-1] == t_h.shape[-1]:
                        s_h_proj = s_h
                    else:
                        s_h_proj = actual_projectors[i](s_h)
                    
                    cos_sim = F.cosine_similarity(s_h_proj.float(), t_h.float(), dim=-1)
                    valid_feat = (1.0 - cos_sim) * attention_mask.to(device)
                    loss_feat += (valid_feat.sum() / num_tokens.to(device))
                    
                loss_feat = loss_feat / len(feat_layers)
                
                # --- 终极融合：强制都在同一张卡上计算 ---
                loss = (
                    lam_log * loss_logit.to(device) + 
                    lam_feat * loss_feat.to(device) + 
                    lam_task * loss_task.to(device)
                )
                
                # 反向传播 (Accelerator 自动处理 Grad Scaling)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_update_step += 1
                    
                    if global_update_step % save_every_n_steps == 0 and accelerator.is_main_process:
                        # 👇 传入 accelerator
                        save_checkpoint(student_model, feat_projectors, optimizer, scheduler, epoch + 1, step, global_update_step, checkpoint_dir, accelerator)

                    if global_update_step >= max_update_steps:
                        accelerator.print("\n🛑 达到最大步数，结束训练！")
                        return accelerator.unwrap_model(student_model)  # ✅ 正确解包
                    # 🛡️ 释放驻留在 CUDA 里的碎片显存
                    if accelerator.sync_gradients and global_update_step % 10 == 0:
                        torch.cuda.empty_cache()
                epoch_loss += loss.item()
            
            # 清理显存
            del t_outputs, s_outputs, s_logits, t_probs, s_log_probs
            
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"  Step [{step}] Upd [{global_update_step}] | Tot Loss: {loss.item():.3f} "
                      f"(Logit: {loss_logit.item():.3f}, Feat: {loss_feat.item():.3f}, CE: {loss_task.item():.3f}) | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # =======================================================
            # 🌟 新增：每 200 步触发一次推理效果大检阅
            # =======================================================
            if global_update_step > 0 and global_update_step % 200 == 0:
                eval_prompts = [
                    "Explain quantum computing in simple terms.",
                    "Write a Python function to sort a list.",
                    "The capital of France is", # 测常识
                    "10 + 25 = " # 测逻辑
                ]
                # 传入 accelerator，函数内部会自动只让主卡运行
                quick_evaluate_generation(student_model, teacher_model, tokenizer, eval_prompts, accelerator)
                
                # 🚨 终极多卡防死锁护盾：
                # GPU 0 刚刚花了几秒钟生成文本，而 GPU 1 和 2 瞬间就跑完了上面的 if 判断
                # 如果不在这里让所有卡等一等，GPU 1 和 2 就会冲进下一次训练导致梯度算错卡死！
                accelerator.wait_for_everyone()
        
    print(f"\n🎉 蒸馏微调结束！总耗时: {(time.time() - t0) / 60:.1f} 分钟")
    student_model.eval()
    torch.cuda.empty_cache()
    
    # 🌟 核心修复 2：将模型脱下 DDP 外衣后，再交还给主程序，防止单卡评测时死锁！
    return accelerator.unwrap_model(student_model)