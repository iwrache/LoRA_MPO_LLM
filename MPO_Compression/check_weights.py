import torch

def tree_inspect():
    path = "/home/roots/xiaoshi/LLM-main/MPO_Compression/healing_checkpoints_chat/checkpoint_upd_1500.pt"
    print(f"🔍 正在用 X 光扫描 Checkpoint 结构...\n")
    
    ckpt = torch.load(path, map_location="cpu")
    
    if not isinstance(ckpt, dict):
        print("❌ 警告：这甚至不是一个字典！")
        return
        
    print("📦 文件的顶层目录结构如下：")
    print("-" * 50)
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f" 📂 文件夹: [{k}] (里面装了 {len(v)} 个元素)")
            # 顺便偷看一下这个文件夹里的前 2 个东西叫什么
            sub_keys = list(v.keys())[:2]
            print(f"      ↳ 偷看里面: {sub_keys} ...")
        elif hasattr(v, "shape"): # 捕捉所有张量类
            print(f" 📄 纯张量: [{k}] (形状: {v.shape})")
        else:
            print(f" 🏷️ 杂物堆: [{k}] (数据类型: {type(v).__name__})")
            
    print("-" * 50)

if __name__ == "__main__":
    tree_inspect()