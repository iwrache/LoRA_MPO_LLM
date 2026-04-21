# calibration.py
import torch
from datasets import load_dataset


def get_activation_scales(model, tokenizer, num_samples=64, max_len=512):
    """
    通过少量真实数据，捕获各个 Linear 层的通道最大激活值 (Outliers)。
    返回一个字典：{ 'layer_name': s_vector_tensor }
    """
    print(f"    -> [Calibration] 开始提取激活异常值，样本数: {num_samples}...")

    # 1. 准备一小批校准数据
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")

    # 防止数据不够长，做个截断保护
    total_tokens = enc.input_ids.shape[1]
    req_tokens = num_samples * max_len
    if total_tokens < req_tokens:
        num_samples = total_tokens // max_len
        print(f"    -> [Calibration] 数据不足，调整样本数为: {num_samples}")

    input_ids = enc.input_ids[0, :num_samples * max_len].view(num_samples, max_len)
    input_ids = input_ids.to(model.device)

    scales = {}
    handles = []

    # 2. 定义 Hook 函数 (升级为 Hessian/RMS 追踪)
    def get_hook(name):
        def hook(module, inp, out):
            # 将输入特征展平为 [Total_Tokens, Hidden_Dim]
            x = inp[0].detach().view(-1, inp[0].shape[-1]).float()
            
            # 🚀 核心升级：计算每个通道特征的平方和 (x^2)
            # 这等价于海森矩阵 (H = X^T X) 的对角线元素！
            x_sq_sum = (x ** 2).sum(dim=0)
            
            # 我们累加总的平方和，并在外层最后处理时取平均和开根号
            # 为了在这个 hook 里保持简单，我们先只累加。
            # 为了不破坏外部已有的逻辑，我们这里依然返回累加后的均方根
            
            num_tokens_in_batch = x.shape[0]
            
            if name not in scales:
                # 初始化：保存 [累加的平方和, 累加的 token 数量]
                scales[name] = [x_sq_sum, num_tokens_in_batch]
            else:
                # 累加：平方和相加，token 数量相加
                scales[name][0] += x_sq_sum
                scales[name][1] += num_tokens_in_batch
        return hook

    # 3. 给模型中所有的 nn.Linear 挂载 Hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(get_hook(name)))

    # 4. 跑一次前向传播（只收集数据，不计算梯度）
    with torch.no_grad():
        batch_size = 4
        for i in range(0, num_samples, batch_size):
            batch = input_ids[i:i+batch_size]
            model(batch)

    # 5. 卸载 Hook，清理现场
    for h in handles:
        h.remove()

    print(f"    -> [Calibration] 成功提取了 {len(scales)} 个 Linear 层的缩放向量。")
    # =================================================================
    # 🚀 在返回前，把累加的平方和计算成最终的 RMS (Hessian 对角线近似)
    # 公式： RMS = sqrt( sum(x^2) / total_tokens + epsilon )
    # =================================================================
    final_scales = {}
    for name, (sq_sum, total_tokens) in scales.items():
        # 除以总 token 数得到均方，然后开根号，加上 1e-8 防止除零
        rms_scale = torch.sqrt(sq_sum / total_tokens + 1e-8)
        final_scales[name] = rms_scale
        
    print(f"    -> [Calibration] 成功提取了 {len(final_scales)} 个 Linear 层的 Hessian (RMS) 缩放向量。")
    return final_scales
