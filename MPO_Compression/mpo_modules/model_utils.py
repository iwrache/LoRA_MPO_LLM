#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Modules - 模型工具

包含模型加载、保存、转换等工具函数。
从 mpo_utils.py 迁移而来。

主要功能：
- make_mpo_from_config: 从配置创建 MPO 层
- load_mpo_model: 加载 MPO 压缩模型
- convert_mpo_to_dense: 将 MPO 转回 dense
- replace_llama_linears: 替换 Llama 线性层为 MPO
- retie_lm_head / check_tied: lm_head 工具
"""

import gc
import glob
import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import load_file as safe_load
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# 从新模块导入
from mpo_modules.core import MPOLinear
from mpo_modules.factorization import factor_linear_mpo, estimate_mpo_bond_dim
from mpo_modules.patches import (
    apply_lm_head_input_fp32,
    apply_rmsnorm_fp32,
    apply_sdpa_safety_patches,
)

# ============================================================
# 辅助工具函数
# ============================================================

# ------------------------------------------------------------
# 重构误差日志函数
# ------------------------------------------------------------


@torch.no_grad()
def _log_reconstruction_error(full_name: str, lin: nn.Linear, mpo: MPOLinear, chi_val: int, core_num: int) -> None:
    """
    计算并打印权重重构误差（MPO vs 原始 dense）

    环境变量:
    - MPO_LOG_RECON: 是否打印重构误差（默认 1）
    """
    try:
        if os.environ.get("MPO_LOG_RECON", "1") == "0":
            return
        # 在 CPU 上构建 dense，减少显存压力
        os.environ.setdefault("MPO_CONTRACT_DEVICE", "cpu")
        W_dense = lin.weight.detach().to(torch.float32).cpu()
        W_mpo = mpo._build_full_weight_fp32().to(torch.float32).cpu()
        rows = int(W_dense.shape[0])
        step = 4096
        diff2 = 0.0
        ref2 = 0.0
        max_diff = 0.0
        max_ref = 0.0
        s = 0
        while s < rows:
            e = min(rows, s + step)
            A = W_dense[s:e, :]
            B = W_mpo[s:e, :]
            D = (A - B).abs()
            diff2 += float((D * D).sum().item())
            ref2 += float((A * A).sum().item())
            md = float(D.max().item())
            ma = float(A.abs().max().item())
            if md > max_diff:
                max_diff = md
            if ma > max_ref:
                max_ref = ma
            s = e
        rel_frob = (diff2**0.5) / (ref2**0.5 + 1e-12) if ref2 > 0 else float("inf")
        rel_max = max_diff / (max_ref + 1e-12) if max_ref > 0 else float("inf")
        print(f"[recon] {full_name}: relF={rel_frob:.3e} relMax={rel_max:.3e} chi={int(chi_val)} cores={int(core_num)}")
    except Exception as e:
        print(f"[recon] {full_name}: error computing reconstruction error: {e}")


@torch.no_grad()
def _report_wx_error(full_name: str, lin: nn.Linear, mpo: MPOLinear) -> None:
    """
    计算并打印输出空间误差 ||WX - W'X||（用随机输入）

    环境变量:
    - MPO_REPORT_WX_ERROR: 是否打印输出误差（默认 0）
    - MPO_WX_SAMPLES: 随机样本数（默认 128）
    """
    try:
        if os.environ.get("MPO_REPORT_WX_ERROR", "0") != "1":
            return
        ns = int(os.environ.get("MPO_WX_SAMPLES", "128"))
        # 取线性层所在设备与 dtype，保证输入与权重 dtype 对齐，避免 matmul/linear dtype mismatch
        try:
            p = next(lin.parameters())
            dev = p.device
            x_dtype = p.dtype
        except StopIteration:
            dev = torch.device("cpu")
            x_dtype = torch.float32
        lin.eval()
        mpo.eval()
        X = torch.randn(ns, int(lin.in_features), dtype=x_dtype, device=dev)
        with torch.no_grad():
            Y0 = lin(X).to(torch.float32)
            Y1 = mpo(X).to(torch.float32)
        diff = Y0 - Y1
        num = float(torch.linalg.norm(diff, ord="fro").item())
        den = float(torch.linalg.norm(Y0, ord="fro").item() + 1e-12)
        rel = num / den
        print(f"[wx_err] {full_name}: ||WX-W'X||={num:.3e} rel={rel:.2%} samples={ns}")
    except Exception as e:
        print(f"[wx_err] {full_name}: failed to compute error: {e}")


# ============================================================
# 模型工具函数
# ============================================================

# ------------------------------------------------------------
# make_mpo_from_config
# ------------------------------------------------------------


def make_mpo_from_config(
    cfg: Dict[str, Any],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> MPOLinear:
    cores: List[torch.Tensor] = []
    for c in cfg["cores"]:
        t = (
            torch.tensor(c, dtype=dtype, device=device)
            if not isinstance(c, torch.Tensor)
            else c.to(device=device, dtype=dtype)
        )
        cores.append(t)

    row_perm = torch.tensor(cfg["row_perm"], dtype=torch.long, device=device) if "row_perm" in cfg else None
    col_perm = torch.tensor(cfg["col_perm"], dtype=torch.long, device=device) if "col_perm" in cfg else None

    inv_row_perm = None
    inv_col_perm = None
    if row_perm is not None:
        inv_row_perm = torch.empty_like(row_perm)
        inv_row_perm[row_perm] = torch.arange(row_perm.size(0), device=device)
    if col_perm is not None:
        inv_col_perm = torch.empty_like(col_perm)
        inv_col_perm[col_perm] = torch.arange(col_perm.size(0), device=device)

    return MPOLinear(
        in_f=cfg["in_f"],
        out_f=cfg["out_f"],
        cores=cores,
        row_perm=row_perm,
        col_perm=col_perm,
        inv_row_perm=inv_row_perm,
        inv_col_perm=inv_col_perm,
    )


# ========================= 递归替换（删去 mpo_error 参数） =========================
def _apply_mpo_placeholders_from_config(
    module: nn.Module,
    mpo_cfg: Dict[str, Any],
    prefix: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    递归地将 nn.Linear 替换为同形状的 MPOLinear 占位符，方便后续加载权重。
    """
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and full in mpo_cfg:
            cfg = mpo_cfg[full]
            shapes = cfg.get("core_shapes", [])
            if not shapes:
                continue
            placeholder_cores: List[torch.Tensor] = [torch.zeros(tuple(shape), dtype=dtype) for shape in shapes]
            mpo_config_dict: Dict[str, Any] = {
                "in_f": child.in_features,
                "out_f": child.out_features,
                "cores": [c.tolist() for c in placeholder_cores],
            }
            if "row_perm" in cfg:
                mpo_config_dict["row_perm"] = cfg["row_perm"]
            if "col_perm" in cfg:
                mpo_config_dict["col_perm"] = cfg["col_perm"]

            mpo_layer = make_mpo_from_config(mpo_config_dict, device=None, dtype=dtype)
            mpo_layer.layer_name = full  # 方便调试打印
            module._modules[name] = mpo_layer
        else:
            _apply_mpo_placeholders_from_config(child, mpo_cfg, full, dtype)


# ------------------------------------------------------------
# load_mpo_model
# ------------------------------------------------------------


def load_mpo_model(
    model_dir: str,
    *,
    device_map: Union[Dict, str, None] = None,
    dtype: torch.dtype = torch.float16,
    dispatch: Union[bool, None] = None,
    load_error: bool = False,
    tokenizer_path: Union[str, None] = None,
    target_device: Union[str, None] = None,  # 新增：直接指定加载到哪个设备
) -> tuple:
    import os

    # 确定目标设备
    # 可通过环境变量 USE_OLD_LOADING=1 强制使用旧的CPU→GPU加载方式
    USE_OLD_LOADING = os.environ.get("USE_OLD_LOADING", "0") == "1"
    if target_device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if USE_OLD_LOADING:
        print(f"[load_mpo_model] Loading MPO model from '{model_dir}' (旧方式: CPU→GPU)")
        load_device = "cpu"  # 先加载到CPU
    else:
        print(f"[load_mpo_model] Loading MPO model from '{model_dir}' → {target_device}")
        load_device = target_device  # 直接加载到目标设备

    # 读 mpo_config.json
    cfg_path = os.path.join(model_dir, "mpo_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"缺少 mpo_config.json: {cfg_path}")
    with open(cfg_path, "r") as f:
        mpo_cfg = json.load(f)
    for k, v in list(mpo_cfg.items()):
        if isinstance(v, (int, float)):
            mpo_cfg[k] = {"chi": int(v), "cores": 2, "core_shapes": []}

    # 聚合 state_dict（直接加载到目标设备）
    def _gather_state_dict_from_dir(weights_dir: str, device: str) -> tuple[dict, bool]:
        sd = {}
        old_key_style = False
        st_idx = os.path.join(weights_dir, "model.safetensors.index.json")
        if os.path.exists(st_idx):
            with open(st_idx, "r") as f:
                idx_json = json.load(f)
            weight_map = idx_json.get("weight_map", {})
            old_key_style = any(".G1" in k for k in weight_map)
            for wf in sorted(set(weight_map.values())):
                sd.update(safe_load(os.path.join(weights_dir, wf), device=device))
            return sd, old_key_style
        pt_idx = os.path.join(weights_dir, "pytorch_model.bin.index.json")
        if os.path.exists(pt_idx):
            with open(pt_idx, "r") as f:
                idx_json = json.load(f)
            weight_map = idx_json.get("weight_map", {})
            old_key_style = any(".G1" in k for k in weight_map)
            for wf in sorted(set(weight_map.values())):
                sd.update(torch.load(os.path.join(weights_dir, wf), map_location=device))
            return sd, old_key_style
        st_single = os.path.join(weights_dir, "model.safetensors")
        if os.path.exists(st_single):
            sd = safe_load(st_single, device=device)
            old_key_style = any(".G1" in k for k in sd.keys())
            return sd, old_key_style
        bin_single = os.path.join(weights_dir, "pytorch_model.bin")
        if os.path.exists(bin_single):
            sd = torch.load(bin_single, map_location=device)
            old_key_style = any(".G1" in k for k in sd.keys())
            return sd, old_key_style
        shards = sorted(glob.glob(os.path.join(weights_dir, "pytorch_model-*.bin")))
        if shards:
            for wf in shards:
                sd.update(torch.load(wf, map_location=device))
            old_key_style = any(".G1" in k for k in sd.keys())
            return sd, old_key_style
        raise FileNotFoundError("找不到权重文件")

    sd, old_key_style = _gather_state_dict_from_dir(model_dir, load_device)

    # 构骨架 & 替换为 MPOLinear
    llama_cfg = LlamaConfig.from_pretrained(model_dir, local_files_only=True)
    if USE_OLD_LOADING:
        # 旧方式：不使用init_empty_weights，直接创建实体模型
        model = LlamaForCausalLM_MPO(llama_cfg)
        _apply_mpo_placeholders_from_config(model.model, mpo_cfg, "model", dtype=torch.float32)
    else:
        # 新方式：使用init_empty_weights避免CPU上的大内存分配
        with init_empty_weights():
            model = LlamaForCausalLM_MPO(llama_cfg)
            _apply_mpo_placeholders_from_config(model.model, mpo_cfg, "model", dtype=torch.float32)

    # 旧命名 → 新命名
    if old_key_style:
        rename = {}
        for old in list(sd.keys()):
            if ".G" in old:
                base, g = old.rsplit(".G", 1)
                try:
                    idx = int(g) - 1
                except Exception:
                    continue
                rename[old] = f"{base}.cores.{idx}"
        for old, new in rename.items():
            if new not in sd:
                sd[new] = sd.pop(old)

    # 加载到目标设备骨架
    if USE_OLD_LOADING:
        # 旧方式：在CPU上加载，然后整体移到GPU
        model = model.to(dtype=dtype)
        res = model.load_state_dict(sd, strict=False)
        print("[load_mpo_model] ✅ 权重已加载到 CPU")
        # 然后移到目标设备
        if target_device != "cpu" and torch.cuda.is_available():
            try:
                model = model.to(target_device)
                print(f"✅ 模型已放到 GPU: {target_device}")
            except RuntimeError:
                print(f"⚠️ 搬到 {target_device} 失败，仍留在 CPU")
    else:
        # 新方式：直接在GPU上创建，避免CPU→GPU搬运
        model = model.to_empty(device=target_device)
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        res = model.load_state_dict(sd, strict=False)
        print(f"[load_mpo_model] ✅ 权重已直接加载到 {target_device}")

    # 🔧 修复: rotary_emb.inv_freq 如果损坏（包含 Inf/NaN/全0），需要重新初始化
    # 注意：始终执行修复，不再支持 SKIP_INV_FREQ_FIX 环境变量
    if hasattr(model.model, "rotary_emb"):
        rope = model.model.rotary_emb
        # 检查 inv_freq 是否损坏：NaN, Inf, 或全 0 (to_empty 的副作用)
        is_corrupted = (
            torch.isinf(rope.inv_freq).any()
            or torch.isnan(rope.inv_freq).any()
            or (rope.inv_freq == 0).all()  # to_empty() 会导致全 0
        )
        if is_corrupted:
            print("[load_mpo_model] ⚠️  rotary_emb.inv_freq 损坏，重新初始化...")
            # 重新计算 inv_freq
            dim = rope.config.hidden_size // rope.config.num_attention_heads
            base = getattr(rope.config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=rope.inv_freq.device) / dim))
            rope.inv_freq = inv_freq
            print("[load_mpo_model] ✅ rotary_emb.inv_freq 已修复")
        elif rope.inv_freq.dtype != torch.float32:
            # 确保是 FP32
            rope.inv_freq = rope.inv_freq.float()
    if getattr(res, "missing_keys", []) or getattr(res, "unexpected_keys", []):
        print(f"[load_mpo_model] Note: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    else:
        print("[load_mpo_model] ✅ state_dict keys all matched.")

    # 已经在 target_device 上了，不需要额外搬运
    # 但如果 device_map 有特殊指定，仍然遵守
    if isinstance(device_map, dict):
        for dev, modules in device_map.items():
            for mname in modules:
                try:
                    dict(model.named_modules())[mname].to(dev)
                except Exception:
                    pass

    # 与训练对齐：关 KV cache
    model.config.use_cache = False

    # attention softmax patch 已移除（deprecated）

    # tokenizer（优先用模型目录）
    if tokenizer_path is not None:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, local_files_only=True)
    else:
        try:
            tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        except Exception:
            parent = os.path.dirname(model_dir.rstrip("/"))
            tok = AutoTokenizer.from_pretrained(parent, use_fast=False, local_files_only=True)
            print(f"[load_mpo_model] ℹ️ tokenizer loaded from parent: {parent}")

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.config.pad_token_id = tok.pad_token_id

    # ===== 可控：是否绑 lm_head <-> embed_tokens（默认不绑；设置 TIE_LM_HEAD=1 才启用） =====
    try:
        tie_env = str(os.environ.get("TIE_LM_HEAD", "0")).strip().lower()
    except Exception:
        tie_env = "0"
    do_tie = tie_env in ("1", "true", "yes", "y", "on")
    if do_tie:
        try:
            # 优先使用封装函数
            retie_lm_head(model)  # out.weight = inp.weight; config.tie_word_embeddings = True
            print("[load_mpo_model] ✅ retied lm_head with embed_tokens (env TIE_LM_HEAD)")
        except Exception:
            # 兜底：手动绑
            try:
                inp = model.get_input_embeddings()
                out = model.get_output_embeddings()
                out.weight = inp.weight
                try:
                    model.config.tie_word_embeddings = True
                except Exception:
                    pass
                print("[load_mpo_model] ✅ retied lm_head with embed_tokens (fallback)")
            except Exception as ee:
                print(f"[load_mpo_model] ❗ retie failed: {repr(ee)}")
        try:
            same = model.lm_head.weight.data.data_ptr() == model.model.embed_tokens.weight.data.data_ptr()
            print("lm_head vs embed_tokens same storage (after retie):", same)
        except Exception:
            pass
    else:
        print("[load_mpo_model] ℹ️ skip retie lm_head (TIE_LM_HEAD not enabled)")

    # SDPA 安全补丁（保持原有）
    apply_sdpa_safety_patches(model)
    # RMSNorm / lm_head 输入 fp32 稳定化（受环境变量控制，默认启用）
    try:
        apply_rmsnorm_fp32(model)
    except Exception:
        pass
    try:
        apply_lm_head_input_fp32(model)
    except Exception:
        pass

    # 默认设置为评测模式（更安全）
    # 如果需要训练，用户可以手动调用 model.train()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, MPOLinear):
            m._cached_weight = None

    return model, tok


# ========================= 可选：保留 dense 转换（方便快速评测） =========================


# ------------------------------------------------------------
# convert_mpo_to_dense
# ------------------------------------------------------------


def convert_mpo_to_dense(model: nn.Module):
    modules = dict(model.named_modules())
    for full_name, mod in list(modules.items()):
        if isinstance(mod, MPOLinear):
            W = mod._build_full_weight_fp32().detach()
            parent = model
            *path, last = full_name.split(".")
            for p in path:
                parent = getattr(parent, p)
            new_lin = nn.Linear(mod.in_f, mod.out_f, bias=False, device=W.device, dtype=W.dtype)
            with torch.no_grad():
                new_lin.weight.copy_(W)
            setattr(parent, last, new_lin)
            if hasattr(mod, "weight_full"):
                delattr(mod, "weight_full")
            print(f"[convert] {full_name}: MPOLinear → Linear, weight={tuple(W.shape)}")


# 请用以下代码替换您 util.py 文件中旧的 factor_linear_mpo 函数


# ------------------------------------------------------------
# replace_llama_linears_by_cfg
# ------------------------------------------------------------


def replace_llama_linears_by_cfg(
    model: nn.Module,
    cfg: Dict[str, Any],
    *,
    skip_mlp: Optional[str] = None,
    activation_scales: Optional[dict] = None, # <--- [新增] 接收校准字典
) -> tuple[nn.Module, dict]:
    """
    Convert a config dict to per-layer chi/core maps and call replace_llama_linears_by_maps.

    cfg keys:
      - mode == 'fixed' (default): 使用 mid_chi / deep_chi
      - mode == 'ratio':  使用 target_ratio 自动估算 bond_dim

    Returns (model, mpo_cfg).
    """
    num_cores = int(cfg.get("num_cores", 3))
    freeze_blocks = int(cfg.get("freeze_blocks", 0))
    mid_blocks = int(cfg.get("mid_blocks", 20))
    if skip_mlp is None:
        skip_mlp = cfg.get("skip_mlp", None)

    num_layers = len(model.model.layers)

    # Build per-layer chi/core maps
    chi_attn: Dict[int, int] = {}
    core_attn: Dict[int, int] = {}
    chi_ffn: Dict[int, int] = {}
    core_ffn: Dict[int, int] = {}

    use_ratio = str(cfg.get("mode", "fixed")).strip().lower() == "ratio"
    if use_ratio:
        target_ratio = float(cfg.get("target_ratio", 0.3))
        deep_ratio = float(cfg.get("deep_ratio", target_ratio))
    else:
        mid_chi = int(cfg.get("mid_chi", 100))
        deep_chi = int(cfg.get("deep_chi", 40))

    for idx in range(num_layers):
        if idx < freeze_blocks:
            continue  # skip frozen blocks

        is_mid = (idx - freeze_blocks) < mid_blocks
        core_attn[idx] = num_cores
        core_ffn[idx] = num_cores

        if use_ratio:
            ratio = target_ratio if is_mid else deep_ratio
            # Attn layers
            for fname in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(model.model.layers[idx].self_attn, fname)
                if isinstance(lin, nn.Linear):
                    chi_attn[idx] = estimate_mpo_bond_dim(
                        lin.in_features, lin.out_features, num_cores, ratio
                    )
                    break
            # FFN layers
            for fname in ("gate_proj", "up_proj", "down_proj"):
                if fname == skip_mlp:
                    continue
                lin = getattr(model.model.layers[idx].mlp, fname)
                if isinstance(lin, nn.Linear):
                    chi_ffn[idx] = estimate_mpo_bond_dim(
                        lin.in_features, lin.out_features, num_cores, ratio
                    )
                    break
        else:
            chi = mid_chi if is_mid else deep_chi
            chi_attn[idx] = chi
            chi_ffn[idx] = chi

    return replace_llama_linears_by_maps(model, chi_attn, core_attn, chi_ffn, core_ffn, skip_mlp=skip_mlp, activation_scales=activation_scales)


# ------------------------------------------------------------
# replace_llama_linears_by_maps
# ------------------------------------------------------------


def replace_llama_linears_by_maps(
    model: nn.Module,
    chi_attn: Dict[int, int],
    core_attn: Dict[int, int],
    chi_ffn: Dict[int, int],
    core_ffn: Dict[int, int],
    *,
    skip_mlp: Optional[str] = None,
    activation_scales: Optional[dict] = None,  # <--- [新增] 接收校准字典
) -> tuple[nn.Module, dict]:
    """
    使用每层 SA/FFN 的 chi/cores map 进行替换。
    返回 (model, mpo_cfg)
    """
    DEFAULT_CORES = 3
    mpo_cfg = {}
    for idx, blk in enumerate(model.model.layers):
        # ==========================================
        # 替换 Self-Attn
        # ==========================================
        if idx in chi_attn:
            chi = int(chi_attn[idx])
            cores = int(core_attn.get(idx, DEFAULT_CORES))
            for fname in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(blk.self_attn, fname)
                if not isinstance(lin, nn.Linear):
                    continue
                before = sum(p.numel() for p in lin.parameters())
                full_name = f"model.layers.{idx}.self_attn.{fname}"
                
                # <--- [新增] 从字典中提取当前层的 s_vector
                s_vector = activation_scales.get(full_name) if activation_scales else None
                
                # <--- [修改] 将 s_vector 传给 MPO 生成函数
                mpo = factor_linear_mpo(
                    lin, 
                    bond_dim=chi, 
                    num_cores=cores, 
                    layer_name=full_name,
                    s_vector=s_vector  # 传入！
                )
                
                _log_reconstruction_error(full_name, lin, mpo, chi, cores)
                _report_wx_error(full_name, lin, mpo)
                if getattr(lin, "bias", None) is not None and getattr(mpo, "bias", None) is not None:
                    with torch.no_grad():
                        mpo.bias.copy_(lin.bias.data)
                setattr(blk.self_attn, fname, mpo)
                after = sum(p.numel() for p in mpo.parameters())
                print(f"  [{idx:02d}] self_attn.{fname:<10}: {before / 1e6:.2f}M → {after / 1e6:.2f}M")
                mpo_cfg[full_name] = {
                    "chi": chi,
                    "cores": cores,
                    "core_shapes": [list(c.shape) for c in mpo.cores],
                }
                
        # ==========================================
        # 替换 FFN
        # ==========================================
        if idx in chi_ffn:
            chi = int(chi_ffn[idx])
            cores = int(core_ffn.get(idx, DEFAULT_CORES))
            for fname in ("gate_proj", "up_proj", "down_proj"):
                if fname == skip_mlp:
                    continue
                lin = getattr(blk.mlp, fname)
                if not isinstance(lin, nn.Linear):
                    continue
                before = sum(p.numel() for p in lin.parameters())
                full_name = f"model.layers.{idx}.mlp.{fname}"
                
                # <--- [新增] 从字典中提取当前层的 s_vector
                s_vector = activation_scales.get(full_name) if activation_scales else None
                
                # <--- [修改] 将 s_vector 传给 MPO 生成函数
                mpo = factor_linear_mpo(
                    lin, 
                    bond_dim=chi, 
                    num_cores=cores, 
                    layer_name=full_name,
                    s_vector=s_vector  # 传入！
                )
                
                _log_reconstruction_error(full_name, lin, mpo, chi, cores)
                if getattr(lin, "bias", None) is not None and getattr(mpo, "bias", None) is not None:
                    with torch.no_grad():
                        mpo.bias.copy_(lin.bias.data)
                setattr(blk.mlp, fname, mpo)
                after = sum(p.numel() for p in mpo.parameters())
                print(f"  [{idx:02d}] mlp.{fname:<10}: {before / 1e6:.2f}M → {after / 1e6:.2f}M")
                mpo_cfg[full_name] = {
                    "chi": chi,
                    "cores": cores,
                    "core_shapes": [list(c.shape) for c in mpo.cores],
                }
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return model, mpo_cfg


# ------------------------------------------------------------
# replace_llama_linears
# ------------------------------------------------------------


def replace_llama_linears(
    model: nn.Module,
    *args,
    skip_mlp: Optional[str] = None,
    activation_scales: Optional[dict] = None, # <--- [新增] 接收校准字典
) -> tuple[nn.Module, dict]:
    """
    兼容入口：
      - replace_llama_linears(model, cfg, ...)
      - replace_llama_linears(model, chi_attn, core_attn, chi_ffn, core_ffn, ...)
    建议直接调用 replace_llama_linears_by_cfg 或 replace_llama_linears_by_maps。
    """
    if len(args) == 1 and isinstance(args[0], dict) and "num_cores" in args[0]:
        return replace_llama_linears_by_cfg(model, args[0], skip_mlp=skip_mlp, activation_scales=activation_scales)
    else:
        chi_attn, core_attn, chi_ffn, core_ffn = args[:4]
        return replace_llama_linears_by_maps(
            model,
            chi_attn,
            core_attn,
            chi_ffn,
            core_ffn,
            skip_mlp=skip_mlp,
            activation_scales=activation_scales
        )


# ---------------------------------------------------------------------------
# 3. LlamaForCausalLM 容器
# ---------------------------------------------------------------------------


# ------------------------------------------------------------
# retie_lm_head
# ------------------------------------------------------------


class LlamaForCausalLM_MPO(LlamaForCausalLM):
    """占位容器：用于在加载时替换线性层为 MPO 版本。"""

    pass


def retie_lm_head(model):
    # 强制让 lm_head.weight 和 embed_tokens.weight 共享同一块存储
    inp = model.get_input_embeddings()  # Llama: model.model.embed_tokens
    out = model.get_output_embeddings()  # Llama: model.lm_head

    # 保险：设备/精度一致
    out.weight = inp.weight  # 关键一步：同一 Parameter 的同一 storage
    model.config.tie_word_embeddings = True  # 标记一下（有些 hf 逻辑会读这个）


# ------------------------------------------------------------
# check_tied
# ------------------------------------------------------------


def check_tied(model):
    emb = model.get_input_embeddings().weight
    lm = model.get_output_embeddings().weight
    same = emb.data_ptr() == lm.data_ptr()
    print(f"lm_head vs embed_tokens same storage: {same}")
    return same


# ---------------------------------------------------------------------------
# Exposed helper for unit testing: factor a single Linear with whitening spec
# ---------------------------------------------------------------------------


# ============================================================
# 导出
# ============================================================

__all__ = [
    "make_mpo_from_config",
    "load_mpo_model",
    "convert_mpo_to_dense",
    "replace_llama_linears_by_cfg",
    "replace_llama_linears_by_maps",
    "replace_llama_linears",
    "retie_lm_head",
    "check_tied",
]
