import os
import torch
import traceback

def test_disk_write(target_dir="/mnt/sx_data"):
    print("="*60)
    print(f" 🔍 启动磁盘读写与重命名权限测试")
    print(f" 📁 目标目录: {target_dir}")
    print("="*60)
    
    # 1. 尝试创建目录
    try:
        if not os.path.exists(target_dir):
            print(f"   -> 目录不存在，尝试创建目录...")
            os.makedirs(target_dir, exist_ok=True)
            print("   ✅ 目录创建成功！")
        else:
            print("   ✅ 目录已存在。")
    except Exception as e:
        print(f"   ❌ 目录创建失败！没有权限。报错: {e}")
        return

    test_file_tmp = os.path.join(target_dir, "progressive_layer_checkpoint.pt.tmp_test")
    test_file_final = os.path.join(target_dir, "progressive_layer_checkpoint.pt_test")
    
    # 构造一个极小的假张量
    dummy_data = {'status': 'success', 'dummy_tensor': torch.randn(10, 10)}

    try:
        # 2. 尝试写入临时文件
        print(f"   -> 尝试写入临时文件: {test_file_tmp} ...")
        torch.save(dummy_data, test_file_tmp)
        print("   ✅ 写入成功！")

        # 3. 尝试重命名文件 (完美模拟阶段 2 结尾的 os.replace 行为)
        print(f"   -> 尝试执行覆盖/重命名操作 (os.replace)...")
        os.replace(test_file_tmp, test_file_final)
        print("   ✅ 重命名成功！")

        # 4. 尝试清理测试文件
        print("   -> 尝试删除测试产生的文件...")
        os.remove(test_file_final)
        print("   ✅ 删除清理成功！")

        print("\n🎉 完美通关！你的脚本对该目录拥有【绝对的完全控制权】。")
        print("🚀 现在你可以放心地启动满血 3 卡训练脚本了，绝对不会再报 cannot be opened！")

    except Exception as e:
        print(f"\n❌ 测试在某一步失败了！")
        print(f"   -> 详细错误栈:")
        traceback.print_exc()
        print("\n💡 诊断建议: ")
        print("如果上面报了 'Permission denied' (权限被拒绝)，请在终端执行以下命令过户房产证：")
        print(f"sudo chown -R $USER:$USER {target_dir}")
        print(f"sudo chmod -R 775 {target_dir}")

if __name__ == "__main__":
    # 如果你想测试别的目录，可以改这里
    test_disk_write("/mnt/sx_data")
    #TY43yd67FC!