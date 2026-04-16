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

    # 2. 定义 Hook 函数
    def get_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()
            x_max = x.abs().view(-1, x.shape[-1]).max(dim=0)[0]
            if name not in scales:
                scales[name] = x_max
            else:
                scales[name] = torch.maximum(scales[name], x_max)
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
    return scales
