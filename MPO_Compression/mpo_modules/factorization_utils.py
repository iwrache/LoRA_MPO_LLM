#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factorization 工具函数 - "边重心轻"拆法与参数量计算

因子分解基础函数 (_find_factors_balanced, _reorder_ofac_ifac) 统一使用
mpo_modules.helpers 中的实现。
"""

import os
from typing import List, Tuple

from mpo_modules.helpers import find_factors_balanced as _find_factors_balanced


def _reorder_ofac_ifac(ofac: List[int], ifac: List[int]) -> Tuple[List[int], List[int]]:
    """
    重排输出和输入因子：out 因子降序、in 因子升序，形成"大×小配对"。
    """
    try:
        of = sorted([int(x) for x in ofac], reverse=True)
        inf = sorted([int(x) for x in ifac])
        return of, inf
    except Exception:
        return ofac, ifac


# ============================================================
# 新的"边重心轻"拆法 - 提高 MLP 的 χ 上限
# ============================================================

# LLaMA-2-7B 维度的"边重心轻"拆法配置
LLAMA2_7B_EDGE_HEAVY_CONFIG = {
    # ============ MLP 层 ============
    # up_proj / gate_proj: 11008 × 4096
    (11008, 4096): {
        "ofac": [688, 1, 16],
        "ifac": [64, 1, 64],
        "chi_max": 1024,
    },
    # down_proj: 4096 × 11008
    (4096, 11008): {
        "ofac": [64, 1, 64],
        "ifac": [688, 1, 16],
        "chi_max": 1024,
    },
    # ============ ATTN 层 ============
    # q/k/v/o_proj: 4096 × 4096
    (4096, 4096): {
        "ofac": [64, 1, 64],
        "ifac": [64, 1, 64],
        "chi_max": 4096,
    },
}


def _find_factors_edge_heavy(
    out_f: int, in_f: int, num_cores: int, layer_name: str = ""
) -> Tuple[List[int], List[int]]:
    """
    "边重心轻"拆法：让 MLP 的 χ 上限从 256 提升到 1024

    核心思路：
    - 均匀拆法的问题：中间 core 的 o×i 太大，导致 χ_max 被压死在 256
    - 新拆法：把中间 core 的 o×i 做小（甚至是 1），让两端 r₀、r₂ 变大
    """
    # 环境变量控制：
    # MPO_MLP_EDGE_HEAVY=1  → MLP 层使用边重心轻 (chi_max: 256 → 1024)
    # MPO_ATTN_EDGE_HEAVY=1 → ATTN 层使用边重心轻 (chi_max: 256 → 4096)
    use_mlp_edge_heavy = os.getenv("MPO_MLP_EDGE_HEAVY", "0") == "1"
    use_attn_edge_heavy = os.getenv("MPO_ATTN_EDGE_HEAVY", "0") == "1"

    # 判断层类型
    is_mlp = ".mlp." in layer_name if layer_name else False
    is_attn = ".self_attn." in layer_name if layer_name else False

    # 根据层类型和环境变量决定是否启用边重心轻
    should_use_edge_heavy = num_cores == 3 and ((is_mlp and use_mlp_edge_heavy) or (is_attn and use_attn_edge_heavy))

    if should_use_edge_heavy:
        key = (out_f, in_f)
        if key in LLAMA2_7B_EDGE_HEAVY_CONFIG:
            config = LLAMA2_7B_EDGE_HEAVY_CONFIG[key]
            return config["ofac"], config["ifac"]

    # 默认使用均匀拆法
    ofac = _find_factors_balanced(out_f, num_cores)
    ifac = _find_factors_balanced(in_f, num_cores)
    ofac, ifac = _reorder_ofac_ifac(ofac, ifac)
    return ofac, ifac


def get_chi_max_for_layer(out_f: int, in_f: int, num_cores: int, layer_name: str = "") -> int:
    """获取指定层的 χ 上限"""
    ofac, ifac = _find_factors_edge_heavy(out_f, in_f, num_cores, layer_name)

    r = [ofac[k] * ifac[k] for k in range(num_cores)]

    if num_cores == 3:
        return min(r[0], r[2])
    elif num_cores == 2:
        return min(r[0], r[1])
    else:
        chi_max = float("inf")
        for s in range(num_cores):
            left = 1
            for j in range(s + 1):
                left *= r[j]
            right = 1
            for j in range(s + 1, num_cores):
                right *= r[j]
            chi_max = min(chi_max, left, right)
        return int(chi_max)


def compute_mpo_params_edge_heavy(out_f: int, in_f: int, chi: int, num_cores: int, layer_name: str = "") -> int:
    """
    计算边重心轻拆法下的 MPO 参数量

    MPO 参数量 = Σ_k r_{k-1} × o_k × i_k × r_k
    对于 3 核：P(χ) = χ(o₀i₀ + o₂i₂) + χ²(o₁i₁)
    """
    ofac, ifac = _find_factors_edge_heavy(out_f, in_f, num_cores, layer_name)

    if num_cores == 3:
        o0i0 = ofac[0] * ifac[0]
        o1i1 = ofac[1] * ifac[1]
        o2i2 = ofac[2] * ifac[2]
        return chi * (o0i0 + o2i2) + chi * chi * o1i1
    else:
        params = 0
        r_prev = 1
        for k in range(num_cores):
            r_k = min(chi, ofac[k] * ifac[k])
            params += r_prev * ofac[k] * ifac[k] * r_k
            r_prev = r_k
        return params


__all__ = [
    "_find_factors_balanced",
    "_reorder_ofac_ifac",
    "_find_factors_edge_heavy",
    "get_chi_max_for_layer",
    "compute_mpo_params_edge_heavy",
    "LLAMA2_7B_EDGE_HEAVY_CONFIG",
]
