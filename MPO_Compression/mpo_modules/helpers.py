#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Utils - 辅助函数模块

包含独立的工具函数：
- 因子分解
- 日志工具
"""

from typing import List

import torch

# ============================================================
# 因子分解工具
# ============================================================


def find_factors_balanced(n: int, num_factors: int) -> List[int]:
    """
    将 n 分解为 num_factors 个乘积因子，尽量平衡。
    用于 MPO 的维度分解。

    Args:
        n: 要分解的数
        num_factors: 因子个数

    Returns:
        因子列表，满足 ∏factors = n

    Example:
        >>> find_factors_balanced(64, 3)
        [4, 4, 4]
        >>> find_factors_balanced(128, 2)
        [8, 16]
    """
    if num_factors == 1:
        return [int(n)]

    factors = []
    d = 2
    temp_n = int(n)

    # 质因数分解
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors.append(d)
            temp_n //= d
        d += 1

    if temp_n > 1:
        factors.append(temp_n)

    # 将质因数分配到 num_factors 个组，尽量平衡
    groups = [1] * num_factors
    for factor in sorted(factors, reverse=True):
        groups.sort()  # 每次选最小的组
        groups[0] *= factor

    return [int(g) for g in groups]


# ============================================================
# 日志工具
# ============================================================


def log_tensor(name: str, tensor: torch.Tensor, *, raise_on_bad: bool = False) -> None:
    """
    轻量的张量健康日志：仅在出现 NaN/Inf 时打印。
    设计目标：让 mpo_modules 自包含，不依赖根目录旧版 `mpo_utils.py`。
    """
    if not torch.is_tensor(tensor):
        return
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    if not (has_nan or has_inf):
        return
    msg = f"[{name}] NaN={has_nan} Inf={has_inf} shape={tuple(tensor.shape)} dtype={tensor.dtype}"
    print(f"⚠️  {msg}")
    if raise_on_bad:
        raise RuntimeError(msg)
