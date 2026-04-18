# healing.py
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets
import random


def quick_evaluate_generation(student_model, teacher_model, tokenizer, prompts):
    """极速对比评估：同时输出压缩后(Student)和压缩前(Teacher)的结果"""
    print(f"\n{'='*60}\n👀 [双轨质量评估] 正在生成 Teacher vs Student 对比文本...\n{'-'*60}")
    
    # 确保两个模型都处于推理模式
    student_model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    
    # 获取两者的首层设备 (兼容多卡 device_map)
    device_s = next(student_model.parameters()).device
    device_t = next(teacher_model.parameters()).device if teacher_model else device_s
    
    for prompt in prompts:
        # 使用最朴素、最安全的问答格式
        chat_prompt = f"Question: {prompt}\nAnswer:"
        
        print(f"👤 Question: {prompt}\n")
        
        # ----------------------------------------------------
        # 👑 1. 跑原版 Teacher (看看“满分答案”长什么样)
        # ----------------------------------------------------
        if teacher_model is not None:
            inputs_t = tokenizer(chat_prompt, return_tensors="pt").to(device_t)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
                t_outputs = teacher_model.generate(
                    **inputs_t, 
                    max_new_tokens=60,
                    do_sample=False, temperature=None, top_p=None,
                    repetition_penalty=1.5, # 强心针：防复读
                    pad_token_id=tokenizer.eos_token_id
                )
            t_text = tokenizer.decode(t_outputs[0][inputs_t.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"👑 [原版 Teacher]: {t_text.strip()}\n")
        
        # ----------------------------------------------------
        # 🤖 2. 跑压缩 Student (看看“术后患者”恢复得怎么样)
        # ----------------------------------------------------
        inputs_s = tokenizer(chat_prompt, return_tensors="pt").to(device_s)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
            s_outputs = student_model.generate(
                **inputs_s, 
                max_new_tokens=60,
                do_sample=False, temperature=None, top_p=None,
                repetition_penalty=1.5, # 强心针：防复读
                pad_token_id=tokenizer.eos_token_id
            )
        s_text = tokenizer.decode(s_outputs[0][inputs_s.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖 [压缩 Student]: {s_text.strip()}\n")
        
        print("-" * 60)
    
    student_model.train() # 评估完切回训练模式
    print("="*60 + "\n")

class WikiCalibrationDataset(Dataset):
    """用于知识恢复的 Wikitext 数据集（纯语言建模，无 loss masking）"""
    def __init__(self, tokenizer, max_samples=20000, max_len=256):
        
        print("\n📚 正在加载 Wiki 数据集 (wikitext-2)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.data = []
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            text = item["text"].strip()
            if len(text) < 50:
                continue
            ids = tokenizer(text, max_length=max_len, truncation=True, 
                          add_special_tokens=True).input_ids
            if len(ids) < 10:
                continue
            input_ids = torch.tensor(ids, dtype=torch.long)
            # Wiki 阶段：所有 token 都计入 loss（标准 LM 目标）
            labels = input_ids.clone()
            self.data.append({"input_ids": input_ids, "labels": labels})
            count += 1
        print(f"✅ 成功构建 {len(self.data)} 条 Wiki 语言建模数据！")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
# ==========================================
# 1. SFT Chat 数据集 (带 Loss Masking)
# ==========================================
class SFTChatDataset(Dataset):
    def __init__(self, tokenizer, max_samples=20000, max_len=256, custom_dataset=None):

        if custom_dataset is not None:
            ds = custom_dataset
        else:
            print("\n📚 正在加载 Chat 数据集 (UltraChat-200k)...")
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        self.data = []
        
        count = 0
        for item in ds:
            if count >= max_samples: 
                break
            messages = item["messages"]
            
            msgs = [m for m in messages if m["role"] != "system"]
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                prompt_str = tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
                
                ans_str = msgs[1]["content"] + tokenizer.eos_token
                ans_ids = tokenizer(ans_str, add_special_tokens=False).input_ids
                
                input_ids = prompt_ids + ans_ids
                # 💡 核心魔法：把 User 的部分打上 -100 掩码！
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
                
        print(f"✅ 成功构建 {len(self.data)} 条带有 Loss Masking 的对话数据！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 在 chat_collate_fn 里，记录每条数据的真实长度
def chat_collate_fn(batch, pad_token_id):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = torch.tensor([len(ids) for ids in input_ids])  # ✅ 新增
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100)
    
    # ✅ 用真实长度构建 mask，不误伤对话内的 </s>
    max_len = input_ids_padded.size(1)
    attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return input_ids_padded, labels_padded, attention_mask.long()

# ==========================================
# 2. 存取逻辑 (专门针对 LoRA 和 TT-Embedding 优化)
# ==========================================
import glob
import os

def save_checkpoint(model, optimizer, scheduler, epoch, step, global_update_step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # 只保存可训练参数，从 14GB 压到 ~100MB
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
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, checkpoint_path)
    
    # 滚动删除，只保留最近 1 个
    existing_ckpts = glob.glob(os.path.join(checkpoint_dir, "checkpoint_upd_*.pt"))
    existing_ckpts.sort(key=os.path.getmtime)
    while len(existing_ckpts) > 1:
        oldest_ckpt = existing_ckpts.pop(0)
        os.remove(oldest_ckpt)
        print(f"🗑️ 已清理过期存档: {oldest_ckpt}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 找不到 checkpoint: {checkpoint_path}")
        return None
    
    print(f"📂 加载 Checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # 兼容新旧格式
    if 'trainable_state_dict' in ckpt:
        trainable_weights = ckpt['trainable_state_dict']
        
        # 💡 核心吐真剂：接住 load_state_dict 的返回值！
        incompatible_keys = actual_model.load_state_dict(trainable_weights, strict=False)
        
        print(f"✅ 在文件里找到了 {len(trainable_weights)} 个可训练参数。")
        
        # 👻 看看有多少个被 PyTorch 拒收了！
        num_rejected = len(incompatible_keys.unexpected_keys)
        print(f"🚨 警告：有 {num_rejected} 个参数因为名字对不上，被拒收了！")
        
        if num_rejected > 0:
            print(f"   👉 拒收的名字长这样 (前3个): {incompatible_keys.unexpected_keys[:3]}")
    else:
        # 旧格式：全量加载
        actual_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    return {
        'epoch': ckpt.get('epoch', 0),
        'global_update_step': ckpt.get('global_update_step', 0),
    }

# ==========================================
# 3. 终极双模型蒸馏训练循环
# ==========================================
def train_healing(student_model, tokenizer, teacher_model=None, dataset_name="chat", 
                  epochs=1, batch_size=1, accum_steps=32, lr=1e-4, seq_len=256,
                  save_every_n_steps=200, resume_from_checkpoint=None, 
                  checkpoint_dir="./healing_checkpoints", max_update_steps=400):
    
    print("\n" + "="*70)
    print("🚀 开始阶段微调: 知识蒸馏 (Teacher-Student KD)")
    print("="*70)
    
    # ------------------------------------------
    # 可以冻结 MPO，只炼 LoRA，或者一起训练
    # ------------------------------------------
    for param in student_model.parameters():
        param.requires_grad = False
        
    trainable_params_count = 0
    lora_params = []
    mpo_params = [] # 新增：存放 MPO 核心参数
    
    for name, param in student_model.named_parameters():
        # 1. 收集 LoRA 捷径
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)
            lora_params.append(param)
            trainable_params_count += param.numel()
        # 2. 收集 MPO 与 TT 核心 (解冻它们！)
        elif "core" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)
            mpo_params.append(param)
            trainable_params_count += param.numel()
                
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"🔓 解冻参数: {trainable_params_count/1e6:.2f}M ({trainable_params_count/total_params:.2%})")

    if teacher_model is None:
        raise ValueError("执行蒸馏必须传入 teacher_model！")

    # 确保 Teacher 完全静默
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # ====================================================
    # 💡 极速修复：贴上 Hugging Face 官方的防刷屏护身符
    # ====================================================
    if hasattr(student_model, "enable_input_require_grads"):
        student_model.enable_input_require_grads()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 准备 DataLoader
    if dataset_name == "wiki":
        dataset = WikiCalibrationDataset(tokenizer, max_samples=20000, max_len=seq_len)
    elif dataset_name == "chat":
        dataset = SFTChatDataset(tokenizer, max_samples=20000, max_len=seq_len)
    elif dataset_name == "mixed":
        # 💡 1. 先拿到我们刚刚调配好的 2万条“混合神药” (生肉)
        raw_mixed_hf_dataset = get_mixed_healing_dataset()
        
        # 💡 2. 扔进你的 SFTChatDataset 加工厂，进行分词和打 Loss 掩码 (煮熟)
        # 注意这里不需要传 max_samples，因为我们在构建函数里已经控制好 20000 条了
        dataset = SFTChatDataset(tokenizer, max_len=seq_len, custom_dataset=raw_mixed_hf_dataset)
    else:
        raise ValueError("暂不支持的数据集！")


    # 准备 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda b: chat_collate_fn(b, tokenizer.pad_token_id))
    
    # ====================================================
    # 💡 修复：必须先计算 feat_layers，再构建投影层！
    # ====================================================
    num_layers = len(student_model.model.layers)
    feat_layers = [i for i in range(0, num_layers, max(1, num_layers // 4))]
    if (num_layers - 1) not in feat_layers:
        feat_layers.append(num_layers - 1)
    print(f"🔍 Feature 蒸馏对齐层级: {feat_layers}")

    # ------------------------------------------
    # 引入 Feature 蒸馏的投影层 (Hint Regression)
    # ------------------------------------------
    hidden_size = student_model.config.hidden_size
    loss_device = next(p.device for p in student_model.parameters() if p.device.type == 'cuda')

    # 为每一个需要对齐的层，建立一个从 Student 映射到 Teacher 空间的线性层
    feat_projectors = nn.ModuleList([
        nn.Linear(hidden_size, hidden_size, bias=False) for _ in feat_layers
    ]).to(loss_device).to(torch.float32) # 保持精度一致

    # 将投影层参数也加入优化器！
    optimizer_grouped_parameters = [
        {"params": lora_params, "lr": lr},
        {"params": mpo_params, "lr": lr * 0.1},
        {"params": feat_projectors.parameters(), "lr": lr} # 👈 投影层跟随主学习率训练
    ]

    # 构建优化器
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, weight_decay=0.01)
    
    # Cosine 衰减 + Warmup
    effective_total = min((len(dataloader) // accum_steps) * epochs, max_update_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(50, int(effective_total * 0.1)), num_training_steps=effective_total
    )
    
    student_model.train()
    device = next(student_model.parameters()).device
    t0 = time.time()
    global_update_step = 0
    start_epoch = 0

    if resume_from_checkpoint is not None:
        resume_info = load_checkpoint(student_model, resume_from_checkpoint, optimizer, scheduler)
        if resume_info is not None:
            start_epoch = 0
            global_update_step = resume_info['global_update_step']  # ← 新增

    # 蒸馏超参数 (按照你的方案配置)
    T = 2.0
    lambda_logit = 1.0
    lambda_feat = 0.1
    lambda_task = 0.5
    
    # ====================================================
    # 💡 修复：DataParallel 安全解包获取真实模型结构
    # ====================================================ƒdel
    # 如果穿着 DP 防弹衣，就拉开拉链 (.module) 进去看，否则直接看
    # 确定 loss 汇聚设备（只算一次）
    loss_device = next(p.device for p in student_model.parameters() if p.device.type == 'cuda')


    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            input_ids, labels, attention_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # ------------------------------------------
            # 1. 👻 获取 Teacher 目标 (全程无梯度)
            # ------------------------------------------
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    # output_hidden_states=True 绝杀！无需繁琐的 Hook！
                    t_outputs = teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # ------------------------------------------
            # 2. 🧠 获取 Student 输出
            # ------------------------------------------
            with torch.amp.autocast('cuda', dtype=torch.float32):
                s_outputs = student_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                loss_task = s_outputs.loss
                
                # ====================================================
                # 💡 修复 1：Top-K Logit 蒸馏 (过滤长尾噪声，专注核心语义)
                # ====================================================
                s_logits = s_outputs.logits / T
                t_logits = t_outputs.logits / T
                
                # 只取 Teacher 的 Top-K (例如 K=128)
                K = 128
                t_topk_vals, t_topk_indices = t_logits.topk(K, dim=-1)
                
                # 构造一个全为负无穷 (-inf) 的张量，并将 top-K 的值精准填入
                mask_t_logits = torch.full_like(t_logits, float('-inf'))
                mask_t_logits.scatter_(dim=-1, index=t_topk_indices, src=t_topk_vals)
                
                # 这样 softmax 后，只有 top-K 的位置有概率（且和为1），其余全为 0
                t_probs = F.softmax(mask_t_logits, dim=-1)
                s_log_probs = F.log_softmax(s_logits, dim=-1)
                
                # 计算 KL 散度 (设备自动对齐 logits 所在显卡)
                kl_loss = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1) 
                
                num_tokens = attention_mask.sum().clamp(min=1)
                
                # 跨卡传送：把 mask 送到 kl_loss 所在的显卡上相乘
                mask_for_kl = attention_mask.to(kl_loss.device)
                
                # 💡 升级 2：焦点蒸馏权重 (Focal KD Weighting)
                with torch.no_grad():
                    # 算一下当前这句话的平均 KL 散度 (基准线)
                    mean_kl = (kl_loss * mask_for_kl).sum() / num_tokens.to(kl_loss.device)
                    # 动态放大：散度越高于平均线，惩罚权重越大 (最高放大 3 倍)
                    # 加上 1e-6 防止除以 0
                    focal_weight = (kl_loss / (mean_kl + 1e-6)).clamp(min=1.0, max=3.0)
                
                # 让原始的 kl_loss 乘以焦点权重！
                valid_kl = kl_loss * focal_weight * mask_for_kl
                loss_logit = (valid_kl.sum() / num_tokens.to(kl_loss.device)) * (T ** 2)
                
                # ====================================================
                # 💡 修复 2：带投影层 (Hint) 的 Feature 蒸馏
                # ====================================================
                loss_feat = torch.tensor(0.0, device=loss_device)
                for i, layer_idx in enumerate(feat_layers):
                    s_h = s_outputs.hidden_states[layer_idx + 1]
                    t_h = t_outputs.hidden_states[layer_idx + 1]
                    
                    # 🚀 将 Student 的特征通过翻译官 (Projector) 映射到 Teacher 的坐标系
                    # 必须统一下发到 loss_device 算！
                    s_h_proj = feat_projectors[i](s_h.to(loss_device))
                    t_h_target = t_h.to(loss_device)
                    
                    # 现在可以在同一个维度空间优雅地算余弦相似度了
                    cos_sim = F.cosine_similarity(s_h_proj, t_h_target, dim=-1)
                    
                    # 跨卡掩码
                    mask_for_feat = attention_mask.to(loss_device)
                    valid_feat = (1.0 - cos_sim) * mask_for_feat
                    
                    layer_loss = (valid_feat.sum() / num_tokens.to(loss_device))
                    loss_feat = loss_feat + layer_loss
                    
                loss_feat = loss_feat / len(feat_layers)
                
                # --- 终极融合（统一汇聚到 loss_device）---
                loss = (lambda_logit * loss_logit.to(loss_device) + 
                        lambda_feat * loss_feat + 
                        lambda_task * loss_task.to(loss_device)) / accum_steps
                
            loss.backward()
            
            del t_outputs, s_outputs, s_logits, t_logits

            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_update_step += 1
                
                if global_update_step > 0 and global_update_step % save_every_n_steps == 0:
                    save_checkpoint(student_model, optimizer, scheduler, epoch + 1, step, global_update_step, checkpoint_dir)

                if global_update_step >= max_update_steps:
                    print(f"\n🛑 已达到阶段最大步数 ({max_update_steps})，结束本阶段训练！")
                    save_checkpoint(student_model, optimizer, scheduler, epoch + 1, step, global_update_step, checkpoint_dir)
                    student_model.eval()
                    return student_model
                
            epoch_loss += loss.item() * accum_steps
        
            if step % 50 == 0:
                print(f"  Step [{step}/{len(dataloader)}] Upd [{global_update_step}] | "
                        f"Tot: {loss.item() * accum_steps:.3f} "
                        f"(Logit: {loss_logit.item():.3f}, Feat: {loss_feat.item():.3f}, CE: {loss_task.item():.3f}) | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}")
                
            # 💡 新增：每 200 步停下来，花 3 秒钟看一眼模型是不是变傻了
            if global_update_step > 0 and global_update_step % 200 == 0:
                eval_prompts = [
                    "Explain quantum computing in simple terms.",
                    "Write a Python function to sort a list.",
                    "The capital of France is", # 测常识
                    "10 + 25 = " # 测逻辑
                ]
                quick_evaluate_generation(student_model, teacher_model, tokenizer, eval_prompts)
        # epoch 结束，打印汇总
        avg_epoch_loss = epoch_loss / (step + 1)
        elapsed = (time.time() - t0) / 60
        print(f"\n📊 Epoch [{epoch+1}/{epochs}] 完成 | "
            f"平均 Loss: {avg_epoch_loss:.4f} | "
            f"累计耗时: {elapsed:.1f} 分钟")
        
    print(f"\n🎉 蒸馏微调结束！总耗时: {(time.time() - t0) / 60:.1f} 分钟")
    student_model.eval()
    torch.cuda.empty_cache()
    return student_model



def get_mixed_healing_dataset(seed=42):
    print("🧪 正在调配大模型黄金康复药剂 (混合数据集)...")

    # ==========================================
    # 💊 第 1 味药：维持情商 (50% - 10,000条)
    # 来源：UltraChat (极其丝滑的日常对话)
    # ==========================================
    print("   -> 抽取 10,000 条纯对话数据 (UltraChat)")
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:10000]")
    # 它本来就是 messages 格式，直接保留这一个核心字段即可
    chat_ds = chat_ds.select_columns(["messages"])

    # ==========================================
    # 💊 第 2 味药：重塑理科大脑 (25% - 5,000条)
    # 来源：GSM8K (高质量小学数学应用题与严密推理)
    # ==========================================
    print("   -> 抽取 5,000 条数学逻辑数据 (GSM8K)")
    math_ds = load_dataset("gsm8k", "main", split="train[:5000]")
    def format_math(example):
        return {"messages": [
            # 💡 换成极其标准的英文指令
            {"role": "user", "content": "Please solve the following math problem and provide a step-by-step reasoning process:\n" + example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]}
    math_ds = math_ds.map(format_math, remove_columns=math_ds.column_names)

    # ==========================================
    # 💊 第 3 味药：修复语法树与抽象逻辑 (15% - 3,000条)
    # 来源：CodeAlpaca (Python/C++ 等代码编写)
    # ==========================================
    print("   -> 抽取 3,000 条代码语法数据 (CodeAlpaca)")
    code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:3000]")
    def format_code(example):
        prompt = example["instruction"]
        if example["input"]:  # 如果有前置输入要求，拼在一起
            prompt += "\n" + example["input"]
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["output"]}
        ]}
    code_ds = code_ds.map(format_code, remove_columns=code_ds.column_names)

    # ==========================================
    # 💊 第 4 味药：填补死记硬背的常识 (10% - 2,000条)
    # 来源：Databricks Dolly-v2 (仅筛选 open_qa 和 general_qa，质量极高)
    # ==========================================
    print("   -> 抽取高质量百科常识数据 (Dolly-v2)")
    # 使用 dolly-v2 替代老旧拉胯的 wiki_qa
    wiki_ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # 💡 核心过滤：只留下百科常识和通用问答，去掉摘要、头脑风暴等杂质
    wiki_ds = wiki_ds.filter(lambda x: x["category"] in ["open_qa", "general_qa"])
    
    safe_count = min(2000, len(wiki_ds)) 
    wiki_ds = wiki_ds.select(range(safe_count))
    
    def format_dolly(example):
        return {"messages": [
            # Dolly 的 instruction 已经是很好的自然语言提问了
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]}
    
    wiki_ds = wiki_ds.map(format_dolly, remove_columns=wiki_ds.column_names)

    # ==========================================
    # 🌪️ 终极融合：放入炼丹炉洗牌
    # ==========================================
    print("🔄 正在将 4 味药材放入炼丹炉并进行强力混合 (Shuffle)...")
    mixed_ds = concatenate_datasets([chat_ds, math_ds, code_ds, wiki_ds])
    
    # 💡 极其重要：必须打乱！否则模型会先学完聊天，再学数学，最后又把聊天忘了
    mixed_ds = mixed_ds.shuffle(seed=seed) 
    
    print(f"✅ 混合神药构建完成！总数据量: {len(mixed_ds)} 条")
    return mixed_ds