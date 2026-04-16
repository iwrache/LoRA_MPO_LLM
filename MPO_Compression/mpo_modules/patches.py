#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Modules - 模型补丁

包含针对 Llama 等模型的各种稳定性补丁和数值安全增强。
从 mpo_utils.py (行 1267-1398, 3628-3800) 迁移而来。
"""

import inspect
import os
import types

import torch
import torch.nn as nn

# ============================================================
# SDPA (Scaled Dot-Product Attention) 安全补丁
# ============================================================


def apply_sdpa_safety_patches(model: nn.Module):
    """
    训练稳定版注意力后端设置，修复 Llama 4D 加性 mask 的数值问题。

    功能：
    1. 设置注意力后端（eager/sdpa）
    2. 修复 4D causal mask 的 float 溢出问题
    3. 关闭 KV cache（训练时）

    环境变量：
        HEAL_ATTN_BACKEND ∈ {'eager','sdpa','sdpa_math'}
            'eager'    : HuggingFace eager 实现（最稳定，推荐训练）
            'sdpa'     : PyTorch SDPA（由 PyTorch 选择具体 kernel）
            'sdpa_math': 标记为 'sdpa'，建议配合 sdpa_kernel(MATH) 使用

        HEAL_RESTORE_GLOBAL_SDPA ∈ {'0','1'}  默认 '1'
            若检测到之前全局替换过 F.scaled_dot_product_attention，则恢复原版

        HEAL_PATCH_MASK ∈ {'0','1'}  默认 '1'
            是否修复 4D 加性 mask 的 float32 + clamp(-1e4,0) 问题

    Example:
        >>> model = LlamaForCausalLM.from_pretrained(...)
        >>> apply_sdpa_safety_patches(model)
        [attn] use eager backend
        🧯 Patched LlamaModel._prepare_4d_causal_attention_mask
    """
    import torch.nn.functional as F

    backend = os.environ.get("HEAL_ATTN_BACKEND", "eager").lower()
    restore_global = os.environ.get("HEAL_RESTORE_GLOBAL_SDPA", "1") == "1"
    do_mask_patch = os.environ.get("HEAL_PATCH_MASK", "1") == "1"

    # 1) 恢复全局 SDPA（如果之前被修改过）
    if restore_global and hasattr(F, "_orig_sdpa"):
        try:
            F.scaled_dot_product_attention = F._orig_sdpa
            delattr(F, "_orig_sdpa")
        except Exception as e:
            print(f"[unpatch] restore global SDPA failed: {e}")

    # 2) 选择注意力实现（通过 HF 配置，不做全局 monkey-patch）
    try:
        if backend == "eager":
            model.config._attn_implementation = "eager"
            model.config.attn_implementation = "eager"
            print("[attn] use eager backend")
        elif backend in ("sdpa", "sdpa_math"):
            model.config._attn_implementation = "sdpa"
            model.config.attn_implementation = "sdpa"
            if backend == "sdpa_math":
                print("[attn] use sdpa backend (hint: wrap with sdpa_kernel(MATH))")
            else:
                print("[attn] use sdpa backend")
        else:
            # 未知值回退到 eager
            model.config._attn_implementation = "eager"
            model.config.attn_implementation = "eager"
            print(f"[attn] unknown HEAL_ATTN_BACKEND='{backend}', fallback to eager")
    except Exception as e:
        print(f"[attn] set backend failed: {e}")

    # 3) Llama 的 4D 加性 mask 修复：float32 + clamp(-1e4,0)
    if do_mask_patch:
        try:
            from transformers.models.llama.modeling_llama import LlamaModel

            _orig_prepare = getattr(LlamaModel, "_prepare_4d_causal_attention_mask", None)
            if _orig_prepare is not None:

                def _safe_prepare_4d(self, attention_mask, *args, **kwargs):
                    mask = _orig_prepare(self, attention_mask, *args, **kwargs)
                    if mask is not None and mask.dtype.is_floating_point:
                        mask = mask.to(torch.float32)
                        mask = torch.clamp(mask, min=-1e4, max=0)
                    return mask

                LlamaModel._prepare_4d_causal_attention_mask = _safe_prepare_4d
                print("🧯 Patched LlamaModel._prepare_4d_causal_attention_mask -> float32 & clamp(-1e4,0)")
        except Exception:
            pass  # mask patch failed, skip

    # 4) 训练对齐：关闭 KV cache
    try:
        model.config.use_cache = False
    except Exception:
        pass


# ============================================================
# RMSNorm FP32 补丁
# ============================================================


def apply_rmsnorm_fp32(model: nn.Module):
    """
    将 LlamaRMSNorm 的前向改为在 float32 下完成归一化与缩放。

    目的：
    - 避免 fp16/bf16 下的数值不稳定
    - 提高训练和推理的数值精度

    环境变量：
        HEAL_RMSNORM_FP32 ∈ {'0','1'}  默认 '1'
            是否启用 RMSNorm FP32 补丁

    Example:
        >>> model = LlamaForCausalLM.from_pretrained(...)
        >>> apply_rmsnorm_fp32(model)
        # 所有 LlamaRMSNorm 模块将在 fp32 下计算
    """
    if os.environ.get("HEAL_RMSNORM_FP32", "1") != "1":
        return

    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
    except Exception:
        return

    count = 0
    for mod in model.modules():
        if isinstance(mod, LlamaRMSNorm) and not hasattr(mod, "_orig_forward_fp32rms"):
            mod._orig_forward_fp32rms = mod.forward

            def _forward(self, x):
                x_dtype = x.dtype
                x32 = x.float()
                var = (x32 * x32).mean(dim=-1, keepdim=True)
                xhat = x32 * torch.rsqrt(var + self.variance_epsilon)
                w32 = self.weight.float()
                y32 = xhat * w32
                return y32.to(x_dtype)

            mod.forward = types.MethodType(_forward, mod)
            count += 1

    if count > 0:
        print(f"🧯 Applied RMSNorm FP32 patch to {count} modules")


# ============================================================
# lm_head FP32 输入补丁
# ============================================================


def apply_lm_head_input_fp32(model: nn.Module):
    """
    在 lm_head 前将 hidden_states 转为 fp32 再做线性映射。

    目的：
    - 避免 fp16/bf16 下 lm_head 的数值溢出
    - 提高 logits 的精度（loss 计算期望 fp32）

    环境变量：
        HEAL_LMHEAD_FP32_INPUT ∈ {'0','1'}  默认 '1'
            是否启用 lm_head FP32 输入补丁

    Example:
        >>> model = LlamaForCausalLM.from_pretrained(...)
        >>> apply_lm_head_input_fp32(model)
        🧯 Applied lm_head FP32 input patch
    """
    if os.environ.get("HEAL_LMHEAD_FP32_INPUT", "1") != "1":
        return

    try:
        lm_head = model.get_output_embeddings()
    except Exception:
        return

    # 若已包裹过则跳过
    if hasattr(lm_head, "_orig_forward_fp32head"):
        return

    lm_head._orig_forward_fp32head = lm_head.forward

    def _forward(self, x):
        import torch.nn.functional as F

        x32 = x.float() if torch.is_tensor(x) else x
        w32 = self.weight.float()
        b32 = self.bias.float() if getattr(self, "bias", None) is not None else None
        y32 = F.linear(x32, w32, b32)
        return y32  # 保持 fp32 logits

    lm_head.forward = types.MethodType(_forward, lm_head)
    print("🧯 Applied lm_head FP32 input patch")


# ============================================================
# 残差守卫（调试用）
# ============================================================


def install_residual_guards(model: nn.Module, guard_layers: int = 6):
    """
    在前 guard_layers 个 block 安装残差守卫（探针）。

    目的：
    - 监控 Attention 和 MLP 的输入/输出
    - 检测 NaN/Inf 并记录日志
    - 可选择性地清洗异常值（当前为探针模式）

    Args:
        model: Llama 模型实例
        guard_layers: 监控的层数

    Note:
        - 默认只记录日志，不修改数值
        - 需要配合 log_tensor 函数使用
        - 生产环境建议关闭

    Example:
        >>> model = LlamaForCausalLM.from_pretrained(...)
        >>> install_residual_guards(model, guard_layers=3)
        # 前 3 层将监控 attention 和 MLP 的输入/输出
    """

    def _guard_tensor(t):
        """清洗异常值（可选）"""
        if torch.isfinite(t).all():
            return t
        t32 = t.float()
        if t.dtype in (torch.float16, torch.bfloat16):
            finfo = torch.finfo(t.dtype)
            t32 = torch.clamp(t32, min=finfo.min, max=finfo.max)
        return torch.nan_to_num(t32, nan=0.0, posinf=0.0, neginf=0.0).to(t.dtype)

    try:
        from mpo_modules.helpers import log_tensor
    except ImportError:

        def log_tensor(name, tensor):
            """简单的日志函数"""
            if not torch.is_tensor(tensor):
                return
            has_nan = torch.isnan(tensor).any()
            has_inf = torch.isinf(tensor).any()
            if has_nan or has_inf:
                print(f"⚠️  [{name}] NaN={has_nan} Inf={has_inf} shape={tuple(tensor.shape)}")

    count = 0
    for li, blk in enumerate(model.model.layers):
        if li >= guard_layers:
            continue

        # --- MLP ---
        prev_mlp = blk.mlp.forward

        def mlp_safe_forward(self, x, *args, **kwargs):
            log_tensor(f"block{li}.mlp.in_x", x)
            out = prev_mlp(x, *args, **kwargs)
            log_tensor(f"block{li}.mlp.out", out)
            return out

        if not hasattr(blk.mlp, "_orig_forward"):
            blk.mlp._orig_forward = prev_mlp
        blk.mlp.forward = types.MethodType(mlp_safe_forward, blk.mlp)

        # --- Attention ---
        prev_attn = blk.self_attn.forward

        def attn_safe_forward(self, hidden_states, *args, **kwargs):
            log_tensor(f"block{li}.attn.in_hidden", hidden_states)

            am = kwargs.get("attention_mask", None)
            if isinstance(am, torch.Tensor):
                try:
                    amin = am.min().item()
                    amax = am.max().item()
                    print(f"[Probe] block{li}.attn.mask: shape={tuple(am.shape)} range=[{amin:.1e},{amax:.1e}]")
                except Exception:
                    pass

            out = prev_attn(hidden_states, *args, **kwargs)

            # 后置探针
            if torch.is_tensor(out):
                log_tensor(f"block{li}.attn.out_hidden", out)
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                log_tensor(f"block{li}.attn.out_hidden", out[0])

            return out

        if not hasattr(blk.self_attn, "_orig_forward"):
            blk.self_attn._orig_forward = prev_attn
        blk.self_attn.forward = types.MethodType(attn_safe_forward, blk.self_attn)

        count += 1

    if count > 0:
        print(f"🧯 Installed residual guards for {count} layers")


# ============================================================
# 移除 Attention 补丁
# ============================================================


def unpatch_llama_attention(model: nn.Module):
    """
    移除之前应用的 Llama 注意力补丁，恢复原始实现。

    逐层恢复 self_attn.forward 到原版：
    - 如果模块有 _orig_forward，则还原
    - 如果 forward 有 __wrapped__，尝试还原
    - 否则跳过

    Example:
        >>> unpatch_llama_attention(model)
        [unpatch] restored block 0 self_attn.forward
        [unpatch] restored block 1 self_attn.forward
        ...
    """
    count = 0
    for i, blk in enumerate(getattr(model, "model", model).layers):
        attn = blk.self_attn

        # 1) 常见：我们自己保存过原函数
        if hasattr(attn, "_orig_forward") and isinstance(attn._orig_forward, types.MethodType):
            attn.forward = attn._orig_forward
            try:
                delattr(attn, "_orig_forward")
            except Exception:
                pass
            print(f"[unpatch] restored block {i} self_attn.forward from _orig_forward")
            count += 1
            continue

        # 2) 如果 forward 被 functools.wraps 包过，会有 __wrapped__
        fw = attn.forward
        orig = getattr(fw, "__wrapped__", None)
        if isinstance(orig, types.FunctionType):
            attn.forward = types.MethodType(orig, attn)
            print(f"[unpatch] restored block {i} self_attn.forward from __wrapped__")
            count += 1
            continue

        # 3) 尝试从闭包里找 orig_forward
        try:
            if hasattr(fw, "__func__"):
                closure = inspect.getclosurevars(fw.__func__)
            else:
                closure = inspect.getclosurevars(fw)

            orig_in_closure = closure.nonlocals.get("orig", None) or closure.nonlocals.get("orig_forward", None)
            if orig_in_closure is not None:
                attn.forward = types.MethodType(orig_in_closure, attn)
                print(f"[unpatch] restored block {i} self_attn.forward from closure")
                count += 1
                continue
        except Exception:
            pass

    if count > 0:
        print(f"[unpatch] restored {count} attention layers")
    else:
        print("[unpatch] no patches found to remove")


# ============================================================
# 辅助函数
# ============================================================


def apply_all_safety_patches(model: nn.Module, **kwargs):
    """
    一键应用所有推荐的安全补丁。

    Args:
        model: Llama 模型实例
        **kwargs: 可选参数
            - skip_sdpa: 跳过 SDPA 补丁
            - skip_rmsnorm: 跳过 RMSNorm 补丁
            - skip_lmhead: 跳过 lm_head 补丁

    Example:
        >>> model = LlamaForCausalLM.from_pretrained(...)
        >>> apply_all_safety_patches(model)
        [attn] use eager backend
        🧯 Applied RMSNorm FP32 patch to 33 modules
        🧯 Applied lm_head FP32 input patch
    """
    if not kwargs.get("skip_sdpa", False):
        apply_sdpa_safety_patches(model)

    if not kwargs.get("skip_rmsnorm", False):
        apply_rmsnorm_fp32(model)

    if not kwargs.get("skip_lmhead", False):
        apply_lm_head_input_fp32(model)

    print("✅ All safety patches applied")


# ============================================================
# 未来扩展
# ============================================================

# 可以添加更多补丁：
# - Qwen 模型的补丁
# - Mistral 模型的补丁
# - 梯度累积的稳定性补丁
# - 混合精度训练的优化
# 等
