#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPO Modules - 核心类

包含 MPOLinear 核心类的完整实现。
从 mpo_utils.py (行 157-1199) 迁移而来。

这是整个 MPO 系统的核心，包含：
- MPOLinear 类（~1043行）
- 多种前向传播路径（dense, stream, classic, cotengra）
- 数值稳定性保护
- 白化支持
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mpo_modules.helpers import log_tensor

# ============================================================
# MPOLinear 核心类
# ============================================================


class MPOLinear(nn.Module):
    """
    Multi-Core MPO 线性层：
    - 训练：默认走 dense（一次还原/分块 matmul），可通过 MPO_CHECKPOINT=1 启用 chunk 级激活 checkpoint 以省显存
    - 推理：如果 MPO_STREAM_FWD=1 且核为 4D (r_prev, o_k, i_k, r_next)，走流式收缩；否则自动回退到 dense
    - 数值稳健：前后都做 finite 清洗与幅度钳位；支持 order 自动纠错（oi/io）
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        cores: List[torch.Tensor],
        s_vector: Optional[torch.Tensor] = None, # <--- [新增] 接收 s_vector
        row_perm: Optional[torch.Tensor] = None,
        col_perm: Optional[torch.Tensor] = None,
        inv_row_perm: Optional[torch.Tensor] = None,
        inv_col_perm: Optional[torch.Tensor] = None,
        order: str = "oi",
        autodetect_order: bool = True,
        cache_on_eval: bool = True,
        boundary: str = "open",
    ):
        super().__init__()
        # ==========================================
        # [新增核心逻辑]：注册逆变换张量
        # ==========================================
        if s_vector is not None:
            # 提前算好逆（1/s），推理时只需做乘法，速度更快
            # 加上 1e-5 防止除以 0
            scale_inv = 1.0 / torch.clamp(s_vector.float(), min=1e-5)
            # 注册为 buffer，这样保存模型权重(state_dict)时会自动带上它
            self.register_buffer("scale_inv", scale_inv)
        else:
            self.register_buffer("scale_inv", None)
            
        self.boundary = boundary
        # Persist boundary flag so it survives state_dict round-trips
        self.register_buffer(
            "_boundary_periodic",
            torch.tensor(1 if boundary == "periodic" else 0, dtype=torch.int8),
        )
        self.in_f, self.out_f = int(in_f), int(out_f)
        # safetensors 要求保存的张量必须是连续内存；统一在注册时保证 contiguous
        self.cores = nn.ParameterList([nn.Parameter(c.contiguous()) for c in cores])
        self.num_cores = len(cores)

        self.register_buffer("row_perm", row_perm)
        self.register_buffer("col_perm", col_perm)
        self.register_buffer("inv_row_perm", inv_row_perm)
        self.register_buffer("inv_col_perm", inv_col_perm)

        self.order = order
        self.autodetect_order = autodetect_order
        self.cache_on_eval = cache_on_eval

        self._cached_weight: Optional[torch.Tensor] = None
        # 反向时清空缓存
        self.register_full_backward_hook(lambda *args: setattr(self, "_cached_weight", None))

        # —— 默认数值稳定性参数（可被环境变量覆盖） ——
        # 流式
        self.default_stream_accum32: bool = True
        self.default_stream_safe_clamp: float = 1e3
        self.default_stream_accum_clamp: float = 1e3
        self.default_stream_check: bool = True
        self.default_stream_early_check: bool = True
        self.default_stream_overflow_guard: float = 0.0
        self.default_stream_balance: bool = True
        self.default_stream_balance_tgt: float = 1e3
        self.default_rest_chunk: int = 512
        self.default_o_step: int = 256
        self.default_o0_chunk: int = 1
        # dense
        self.default_dense_input_clamp: float = 0.0
        self.default_dense_safe_clamp: float = 1e3

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Sync self.boundary from the persisted buffer after loading
        buf = getattr(self, "_boundary_periodic", None)
        if buf is not None:
            self.boundary = "periodic" if int(buf.item()) else "open"

    # ---------- 通用小工具 ----------
    @staticmethod
    def _has_nan_inf(x: torch.Tensor) -> bool:
        return not torch.isfinite(x).all().item()

    @staticmethod
    def _clean_finite(x: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(x).all():
            return x
        x32 = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype.is_floating_point:
            finfo = torch.finfo(x.dtype)
            x32 = torch.clamp(x32, min=finfo.min, max=finfo.max)
        return x32.to(x.dtype)

    # ---------- 4D 核链一次性还原（dense 权重构建） ----------
    def _contract_4d_chain_with(self, cores: List[torch.Tensor], order: str) -> torch.Tensor:
        """
        cores: [r_{k-1}, o_k, i_k, r_k] 形状的列表
        order: "oi"（先排所有 o，再排所有 i）或 "io"
        返回: [out_f, in_f] 的 dense 权重
        """
        # 逐核 tensordot 收缩虚 bond（r_k）
        x = cores[0]
        for c in cores[1:]:
            # 收缩最后一维(-1=r_k-1)与下一核第一维(0=r_k-1)
            x = torch.tensordot(x, c, dims=([-1], [0]))

        # 去掉首尾可能为 1 的 bond 维
        if x.shape[0] == 1:
            x = x[0]
        if x.shape[-1] == 1:
            x = x[..., 0]

        num = len(cores)  # 核数
        # 此时 x 的轴交替为 (o1,i1,o2,i2,...,on,in)
        if order == "oi":
            # 先取所有奇数位(从0开始的偶位：o轴)，再取所有偶数位(奇位：i轴)
            perm = list(range(0, 2 * num, 2)) + list(range(1, 2 * num, 2))
            x = x.permute(*perm).contiguous()
            o_shape = x.shape[:num]
            i_shape = x.shape[num:]
            W = x.view(int(torch.tensor(o_shape).prod()), int(torch.tensor(i_shape).prod()))
        else:  # "io"
            perm = list(range(1, 2 * num, 2)) + list(range(0, 2 * num, 2))
            x = x.permute(*perm).contiguous()
            i_shape = x.shape[:num]
            o_shape = x.shape[num:]
            W = x.view(int(torch.tensor(i_shape).prod()), int(torch.tensor(o_shape).prod())).t()
        return W

    def _build_full_weight_fp32(self) -> torch.Tensor:
        """
        把 MPO 核还原成 dense 权重（fp32）。2D 核当作链式矩阵相乘。
        - 自动尝试 order 纠错：如果当前 order 产生的 W 异常（非有限/形状对不上/幅度爆炸），
          再尝试交替 order，并选择更健康的那一个。
        """
        contract_dev = os.environ.get("MPO_CONTRACT_DEVICE", "gpu").lower()
        if contract_dev == "cpu":
            tgt_dev = "cpu"
        else:
            try:
                tgt_dev = next(self.parameters()).device
            except StopIteration:
                tgt_dev = "cpu"

        cores32 = [c.to(device=tgt_dev, dtype=torch.float32) for c in self.cores]

        # 2D 情况：按链相乘
        if all(c.ndim == 2 for c in cores32):
            W = cores32[0]
            for c in cores32[1:]:
                W = W @ c
            if W.shape != (self.out_f, self.in_f):
                W = W.view(self.out_f, self.in_f)
            return self._clean_finite(W)

        # 4D 情况
        if self.boundary == "periodic":
            from mpo_modules.ring_ops import contract_ring

            W = contract_ring(cores32, self.order)
        else:
            W = self._contract_4d_chain_with(cores32, self.order)

        def _bad(t: torch.Tensor) -> bool:
            if t.shape != (self.out_f, self.in_f):
                return True
            if not torch.isfinite(t).all():
                return True
            # 简单幅度健康度
            mx = t.abs().max().item()
            return not (mx < 1e6)

        if self.autodetect_order and _bad(W):
            alt = "io" if self.order == "oi" else "oi"
            if self.boundary == "periodic":
                from mpo_modules.ring_ops import contract_ring
                W2 = contract_ring(cores32, alt)
            else:
                W2 = self._contract_4d_chain_with(cores32, alt)
            cand = []
            for tag, M in (("cur", W), ("alt", W2)):
                if M.shape != (self.out_f, self.in_f) or not torch.isfinite(M).all():
                    score = float("inf")
                else:
                    score = M.abs().max().item() + 0.1 * M.abs().mean().item()
                cand.append((score, tag, M))
            cand.sort(key=lambda x: x[0])
            W = cand[0][2]

        if W.shape != (self.out_f, self.in_f):
            W = W.view(self.out_f, self.in_f)
        return self._clean_finite(W)

    # ---------- 流式路径必要的形状因子 ----------
    def _init_stream_factors(self) -> bool:
        """
        检查是否满足 4D MPO（r_prev, o_k, i_k, r_next），并记录各个 (i_k, o_k) 因子。
        """
        if getattr(self, "_stream_ready", False):
            return True
        ins, outs, bonds = [], [], []
        for c in self.cores:
            if c.ndim != 4:
                self._stream_ready = False
                return False
            rp, ok, ik, rn = map(int, c.shape)
            ins.append(ik)
            outs.append(ok)
            bonds.append((rp, rn))
        if self.boundary == "periodic":
            # Stream path not yet implemented for periodic boundary; fall back to dense.
            self._stream_ready = False
            return False
        else:
            if bonds[0][0] != 1 or bonds[-1][1] != 1:
                self._stream_ready = False
                return False
        import math as _math

        if _math.prod(ins) != int(self.in_f) or _math.prod(outs) != int(self.out_f):
            self._stream_ready = False
            return False
        self._in_factors = tuple(ins)
        self._out_factors = tuple(outs)
        self._stream_ready = True
        return True

    # ---------- 一次性搬运/缓存 cores（减小循环内 .to 开销） ----------
    def _prepare_stream_cores(self, device, dtype):
        """
        为 STREAM/ctg/cuQuantum 路径准备 cores。

        约定：
          - 训练阶段（self.training=True）默认 **不缓存**，每次从 Parameter 重新构建张量，
            确保 autograd 图与 self.cores 正确连接，适合反向传播和优化。
          - 推理阶段默认启用缓存（可通过环境变量 MPO_STREAM_CACHE=0 关闭），
            以避免重复的 .to(device,dtype) 开销。
        """
        import os as _os

        # 训练阶段一律不缓存，确保梯度从 cores_cached 回传到参数 self.cores
        cache_default = not self.training
        use_cache = _os.getenv("MPO_STREAM_CACHE", "1" if cache_default else "0")
        use_cache = use_cache.strip().lower() not in ("0", "false", "no", "n", "off")

        key = (device, dtype)
        if use_cache and getattr(self, "_cores_cached_key", None) == key and hasattr(self, "_cores_cached"):
            return self._cores_cached

        cores = []
        for c in self.cores:
            if c.device == device and c.dtype == dtype:
                t = c
            else:
                t = c.to(device=device, dtype=dtype, non_blocking=True)
            cores.append(t)

        if use_cache:
            self._cores_cached = tuple(cores)
            self._cores_cached_key = key
        return tuple(cores)

    # ---------- 流式前向（推理友好；训练慎用） ----------
    def _forward_stream_contract(self, x: torch.Tensor) -> torch.Tensor:
        """
        流式前向（4D MPO：core 形状 [a=o_{k-1}, o_k, b=i_k, c=o_k]）：
        - R 轴（rest/in-side）与 o 轴双分块，控制中间态规模；
        - 每个核入口只 permute 一次 core -> [a,b,o,c]，减少重复开销；
        - 每个 R 分块只构建一次 y_mat，多个 o 子分块复用；
        - 训练时使用"分片累加"（保持可导），推理时使用 `copy_`（更快）。

        环境变量（可选）：
        MPO_STREAM_REST_CHUNK : R_next 的分块大小（默认 4096）
        MPO_STREAM_O_STEP     : o 轴子分块大小（默认 2048）
        MPO_STREAM_O0_CHUNK   : 首核 o1 的 tile 大小（默认 8）
        MPO_STREAM_ACCUM32    : 在流式中间态使用 float32 累加（默认 0=关闭）
        MPO_STREAM_SAFE_CLAMP : 半精度输出的安全幅度钳位（默认 1e4）
        MPO_STREAM_CHECK      : 结束后检查非有限值，发现则触发回退（默认 1）
        MPO_DEBUG             : '1' 打印关键路径与分块信息
        """
        import os

        debug = os.getenv("MPO_DEBUG", "0") == "1"
        # —— 可开关的函数级 profiling（record_function & NVTX） ——
        layer_name = getattr(self, "layer_name", "(unknown)")
        prof_on = os.getenv("MPO_PROFILE", "0") == "1"
        nvtx_on = os.getenv("MPO_NVTX", "0") == "1"

        class _NoopCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def _prof(name: str):
            if not prof_on and not nvtx_on:
                return _NoopCtx()
            rf = torch.autograd.profiler.record_function(name) if prof_on else _NoopCtx()

            class _Ctx:
                def __enter__(self):
                    # 进入 record_function
                    self._rf = rf.__enter__() if hasattr(rf, "__enter__") else None
                    # 推 NVTX range（若开启）
                    if nvtx_on:
                        try:
                            torch.cuda.nvtx.range_push(name)
                        except Exception:
                            pass
                    return self

                def __exit__(self, et, ev, tb):
                    # 弹 NVTX range（若开启）
                    if nvtx_on:
                        try:
                            torch.cuda.nvtx.range_pop()
                        except Exception:
                            pass
                    # 退出 record_function
                    if hasattr(rf, "__exit__"):
                        rf.__exit__(et, ev, tb)
                    return False

            return _Ctx()

        # —— 环境变量读取小工具 ——
        def _get_env_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, None)
            if val is None:
                return bool(default)
            v = val.strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            try:
                return bool(int(v))
            except Exception:
                return bool(default)

        def _get_env_int(key: str, default: int) -> int:
            val = os.getenv(key, None)
            if val is None:
                return int(default)
            try:
                return int(val)
            except Exception:
                return int(default)

        def _get_env_float(key: str, default: float) -> float:
            val = os.getenv(key, None)
            if val is None:
                return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)

        # 统一使用代码内默认值（可被环境变量覆盖）
        use_accum32 = _get_env_bool("MPO_STREAM_ACCUM32", self.default_stream_accum32)
        safe_clamp = _get_env_float("MPO_STREAM_SAFE_CLAMP", self.default_stream_safe_clamp)
        do_check = _get_env_bool("MPO_STREAM_CHECK", self.default_stream_check)
        early_check = _get_env_bool("MPO_STREAM_EARLY_CHECK", self.default_stream_early_check)
        overflow_guard = _get_env_float("MPO_STREAM_OVERFLOW_GUARD", self.default_stream_overflow_guard)
        accum_clamp = _get_env_float("MPO_STREAM_ACCUM_CLAMP", self.default_stream_accum_clamp)
        balance_on = _get_env_bool("MPO_STREAM_BALANCE", self.default_stream_balance)
        balance_tgt = _get_env_float("MPO_STREAM_BALANCE_TGT", self.default_stream_balance_tgt)
        # 总开关：关闭则跳过所有数值保护与平衡
        guard_on = _get_env_bool("MPO_STREAM_GUARD", False)
        if not guard_on:
            do_check = False
            early_check = False
            overflow_guard = 0.0
            accum_clamp = 0.0
            balance_on = False

        # —— 输入清洗（避免 NaN/Inf 传入，兼容半精度）——
        x_in = x
        if guard_on:
            x = self._clean_finite(x)
        device = x.device
        in_dt = x.dtype
        accum_dt = torch.float32 if use_accum32 else in_dt

        # 若还未初始化形状因子，这里补一次（正常在 forward 外层已保证）
        if not getattr(self, "_stream_ready", False):
            ok = self._init_stream_factors()
            if not ok:
                raise RuntimeError("[MPOLinear] stream path requires 4D cores with consistent factors")

        # 预搬运/缓存后的 cores，accum32 时直接准备为 fp32，避免中途混用半精度
        with _prof(f"MPOLinear[{layer_name}].stream.prepare_cores"):
            cores = self._prepare_stream_cores(device, torch.float32 if use_accum32 else in_dt)

        # 维度信息
        B_dims = x.shape[:-1]  # 批维（可能为空）
        in_f = int(self.in_f)
        out_f = int(self.out_f)
        in_factors = list(self._in_factors)  # [i1, i2, ..., id]
        out_factors = list(self._out_factors)  # [o1, o2, ..., od]
        d = len(in_factors)

        # 分块参数（支持环境变量覆盖）
        REST_CHUNK = _get_env_int("MPO_STREAM_REST_CHUNK", self.default_rest_chunk)
        O_STEP = _get_env_int("MPO_STREAM_O_STEP", self.default_o_step)
        O0_CHUNK = _get_env_int("MPO_STREAM_O0_CHUNK", self.default_o0_chunk)

        # 输出扁平映射：o_d 最快变，o_1 最慢变
        stride_tail = 1
        for v in out_factors[1:]:
            stride_tail *= int(v)

        # 预分配最终输出（accum dtype 可选 fp32）
        with _prof(f"MPOLinear[{layer_name}].stream.alloc_y_out"):
            y_out = torch.zeros(*B_dims, out_f, device=device, dtype=accum_dt)

        # 初始状态：(*B, a=1, R_flat=in_f, P=1)
        with _prof(f"MPOLinear[{layer_name}].stream.init_base_state"):
            base_state = (
                x.reshape(*B_dims, in_f).view(*B_dims, 1, in_f, 1).contiguous()
                if not use_accum32
                else x.float().reshape(*B_dims, in_f).view(*B_dims, 1, in_f, 1).contiguous()
            )
            # 入口清洗/钳位（可关）
            if guard_on:
                base_state = self._clean_finite(base_state)
                if accum_clamp > 0:
                    base_state = torch.clamp(base_state, min=-accum_clamp, max=accum_clamp)

        # 首核 o1 维 tile
        o1 = int(out_factors[0])
        tile = max(1, O0_CHUNK)

        if debug:
            phase = "train" if self.training else "eval"
            print(
                f"[MPOLinear:{getattr(self, 'layer_name', '(unknown)')}] STREAM "
                f"phase={phase} o1={o1} tile={tile} "
                f"REST_CHUNK={REST_CHUNK} O_STEP={O_STEP}"
            )

        for o1_s in range(0, o1, tile):
            o1_e = min(o1, o1_s + tile)

            # 每个 tile 重置状态
            y_state = base_state
            R_flat = in_f
            P_prev = 1

            for k in range(d):
                core = cores[k]  # [a, o_k, b, c]
                a_sz, o_sz, b_sz, c_sz = map(int, core.shape)

                # 首核仅切 o1 段，其它核走全量 o
                if k == 0:
                    o_start, o_end = o1_s, o1_e
                else:
                    o_start, o_end = 0, o_sz

                # 一致性检查
                if int(y_state.shape[-3]) != a_sz:
                    raise RuntimeError(f"[MPOLinear] r_prev mismatch at core#{k}: {int(y_state.shape[-3])} vs {a_sz}")
                if R_flat % b_sz != 0:
                    raise RuntimeError(f"[MPOLinear] rest_flat {R_flat} not divisible by i_k={b_sz} at core#{k}")

                R_next = R_flat // b_sz

                # 只在核入口做一次 permute：core → [a,b,o,c]
                with _prof(f"MPOLinear[{layer_name}].stream.core{k}.permute"):
                    core_aboc = core.permute(0, 2, 1, 3).contiguous()

                # 为下一个状态分配张量：(*B, c, R_next, P_next_total)
                P_next_total = P_prev * (o_end - o_start)
                with _prof(f"MPOLinear[{layer_name}].stream.core{k}.alloc_next_state"):
                    next_state = torch.zeros(*B_dims, c_sz, R_next, P_next_total, device=device, dtype=accum_dt)

                # —— R 轴分块 —— #
                r_step = max(1, min(REST_CHUNK, R_next))
                for r_s in range(0, R_next, r_step):
                    r_e = min(R_next, r_s + r_step)
                    Rb = r_e - r_s

                    # y_blk: (*B, a, b, Rb, P_prev)
                    # 先 reshape 再选 Rb 段
                    with _prof(f"MPOLinear[{layer_name}].stream.core{k}.R[{r_s}:{r_e}].reshape_select"):
                        y_blk = y_state.reshape(*B_dims, a_sz, b_sz, R_next, P_prev)[..., :, :, r_s:r_e, :]

                    # y_mat: (*B, Rb, P_prev, ab)
                    with _prof(f"MPOLinear[{layer_name}].stream.core{k}.R[{r_s}:{r_e}].prep_mat"):
                        y_mat = (
                            y_blk.permute(
                                *range(len(B_dims)),  # 保留批维在前
                                -2,  # Rb
                                -1,  # P_prev
                                -4,  # a
                                -3,  # b
                            )
                            .contiguous()
                            .view(*B_dims, Rb, P_prev, a_sz * b_sz)
                        )

                    # —— o 轴子分块 —— #
                    o_step = max(1, min(O_STEP, o_end - o_start))
                    o_cur = o_start
                    while o_cur < o_end:
                        o_e = min(o_end, o_cur + o_step)
                        o_sub = o_e - o_cur

                        # 在 core_aboc 上直接切片并展平到 (ab, o_sub*c)
                        with _prof(f"MPOLinear[{layer_name}].stream.core{k}.slice_flatten"):
                            core_slice = core_aboc[:, :, o_cur:o_e, :]  # [a,b,o_sub,c]
                            core_flat = core_slice.reshape(a_sz * b_sz, o_sub * c_sz)  # [ab, o_sub*c]

                        # (*B, Rb, P_prev, ab) @ (ab, o_sub*c) → (*B, Rb, P_prev, o_sub*c)
                        # 数值平衡：可选对 (y_mat, core_flat) 做配对缩放，保持乘积不变但抑制幅度
                        if balance_on:
                            try:
                                amax = y_mat.abs().amax()
                                wmax = core_flat.abs().amax()
                                # 只在超过阈值时触发缩放
                                if bool((amax > balance_tgt) or (wmax > balance_tgt)):
                                    if amax >= wmax:
                                        s = (amax / balance_tgt).clamp(min=1.0)
                                        y_eff = y_mat / s
                                        w_eff = core_flat * s
                                    else:
                                        s = (wmax / balance_tgt).clamp(min=1.0)
                                        y_eff = y_mat * s
                                        w_eff = core_flat / s
                                else:
                                    y_eff = y_mat
                                    w_eff = core_flat
                            except Exception:
                                y_eff = y_mat
                                w_eff = core_flat
                        else:
                            y_eff = y_mat
                            w_eff = core_flat

                        # 为稳定性：在 fp32 中做 GEMM；累加阶段可选保持 fp32
                        with _prof(f"MPOLinear[{layer_name}].stream.core{k}.R[{r_s}:{r_e}].O[{o_cur}:{o_e}].gemm"):
                            z2d32 = torch.matmul(y_eff.float(), w_eff.float())
                        z_tmp = z2d32.to(accum_dt).view(*B_dims, Rb, P_prev, o_sub, c_sz)  # (*B, Rb, P_prev, o_sub, c)

                        # 变换为：(*B, c, Rb, P_prev, o_sub)
                        with _prof(f"MPOLinear[{layer_name}].stream.core{k}.post_gemm.permute"):
                            z_blk = z_tmp.permute(
                                *range(len(B_dims)),
                                -1,  # c
                                -4,  # Rb
                                -3,  # P_prev
                                -2,  # o_sub
                            ).contiguous()

                        # 压平成 P 维：(*B, c, Rb, P_prev*o_sub)
                        with _prof(f"MPOLinear[{layer_name}].stream.core{k}.post_gemm.transform"):
                            z_flat = z_blk.view(*B_dims, c_sz, Rb, P_prev * o_sub)

                        # 中间结果清洗/钳位/早期检查（可关）
                        if guard_on:
                            z_flat = self._clean_finite(z_flat)
                            if accum_clamp > 0:
                                z_flat = torch.clamp(z_flat, min=-accum_clamp, max=accum_clamp)
                            if early_check and (not torch.isfinite(z_flat).all()):
                                raise RuntimeError("non-finite in z_flat")
                            if overflow_guard > 0:
                                if z_flat.abs().max().item() > overflow_guard:
                                    raise RuntimeError("overflow guard in z_flat")

                        # 写入 next_state 的正确 P 段
                        p_s = P_prev * (o_cur - o_start)
                        p_e = p_s + P_prev * o_sub

                        if self.training:
                            # 训练：累加可导
                            with _prof(f"MPOLinear[{layer_name}].stream.core{k}.write.add"):
                                next_state[..., :, r_s:r_e, p_s:p_e] = next_state[..., :, r_s:r_e, p_s:p_e] + z_flat
                        else:
                            # 推理：直接 copy_ 更快
                            with _prof(f"MPOLinear[{layer_name}].stream.core{k}.write.copy"):
                                next_state[..., :, r_s:r_e, p_s:p_e].copy_(z_flat)

                        # 释放临时引用，加速显存回收
                        del core_slice, core_flat, z_tmp, z_blk, z_flat
                        o_cur = o_e

                    del y_blk, y_mat  # 每个 R 分块结束后释放

                # 进入下一核前，稳定 next_state（可关）
                if guard_on:
                    next_state = self._clean_finite(next_state)
                    if accum_clamp > 0:
                        next_state = torch.clamp(next_state, min=-accum_clamp, max=accum_clamp)
                    if early_check and (not torch.isfinite(next_state).all()):
                        raise RuntimeError("non-finite in next_state")
                    if overflow_guard > 0:
                        if next_state.abs().max().item() > overflow_guard:
                            raise RuntimeError("overflow guard in next_state")

                # 进入下一核
                y_state = next_state  # (*B, c, R_next, P_next_total)
                P_prev = P_next_total
                R_flat = R_next
                del next_state, core, core_aboc

            # d 个核结束：y_state 形如 (*B, 1, 1, P_tile)
            y_tile = y_state.squeeze(-3).squeeze(-2).contiguous()  # (*B, P_tile)

            # 写回输出连续切片（o1 最慢变）
            start = o1_s * stride_tail
            end = o1_e * stride_tail
            if self.training:
                y_out[..., start:end] = y_out[..., start:end] + y_tile
            else:
                y_out[..., start:end].copy_(y_tile)
            del y_tile

        # 输出清洗与安全钳位（可关）
        if guard_on:
            y_out = self._clean_finite(y_out)
            # 可选：结束检查（用于触发回退）
            if do_check and (not torch.isfinite(y_out).all()):
                raise RuntimeError("non-finite detected in stream output")

        tgt = x_in.dtype
        # 更保守的幅度钳位：半精度用 [-safe_clamp, safe_clamp]
        if guard_on and tgt in (torch.float16, torch.bfloat16):
            y_out = torch.clamp(y_out, min=-safe_clamp, max=safe_clamp)
        return y_out.to(tgt)

    def _forward_mpo_classic(self, x: torch.Tensor) -> torch.Tensor:
        """
        经典 MPO 收缩（无分块版，纯 tensordot 链）：
        - 不构建 dense 权重，按 4D MPO 链顺序一次性进行收缩；
        - 与流式实现相同的数学形式，但移除 R/o 轴分块循环；
        - 显存占用与 out_f 成正比，适合做基准或在显存足够时使用。
        - 不调用 cotengra；如需 ctg，请在上层调度选择 _forward_mpo_cotengra。
        """
        import os

        # 环境开关（与流式保持一致）
        def _get_env_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, None)
            if val is None:
                return bool(default)
            v = val.strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            try:
                return bool(int(v))
            except Exception:
                return bool(default)

        def _get_env_float(key: str, default: float) -> float:
            val = os.getenv(key, None)
            if val is None:
                return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)

        use_accum32 = _get_env_bool("MPO_STREAM_ACCUM32", self.default_stream_accum32)
        safe_clamp = _get_env_float("MPO_STREAM_SAFE_CLAMP", self.default_stream_safe_clamp)
        guard_on = _get_env_bool("MPO_STREAM_GUARD", False)

        # 形状准备
        if not getattr(self, "_stream_ready", False):
            ok = self._init_stream_factors()
            if not ok:
                # 若严格模式，拒绝回退
                if os.getenv("MPO_STRICT_STREAM", "0") == "1":
                    raise RuntimeError("[MPOLinear] strict stream requires 4D MPO cores; fallback to dense is disabled")
                # 不满足 4D MPO，回退到 dense
                W32 = self._build_full_weight_fp32()
                x32 = x.to(torch.float32)
                y32 = x32 @ W32.t()
                return y32.to(x.dtype)

        device = x.device
        in_dt = x.dtype
        accum_dt = torch.float32 if use_accum32 else in_dt

        B_dims = x.shape[:-1]
        in_f = int(self.in_f)
        out_factors = list(self._out_factors)  # [o1, o2, ..., od]
        in_factors = list(self._in_factors)  # [i1, i2, ..., id]
        d = len(in_factors)

        # （取消 d==3 特化，统一走通用 tensordot 链实现）

        # —— 通用：任意 d 的 tensordot 链收缩 ——
        # y 形状：(*B, i1, i2, ..., id, r_prev=1)
        y = x.reshape(*B_dims, *in_factors)
        y = y.unsqueeze(y.ndim)  # append r_prev=1 as last axis
        if guard_on:
            y = self._clean_finite(y)

        # 追踪各 i 轴位置（动态）
        b = len(B_dims)
        i_axes = list(range(b, b + d))
        r_prev_idx = y.ndim - 1

        for k in range(d):
            core = self.cores[k]  # [r_{k-1}, o_k, i_k, r_k]
            # 当前待收缩的 i 轴是 i_axes[0]
            i_idx = i_axes[0]
            # tensordot 收缩 r_prev 与 i_k
            a = y.float() if use_accum32 else y
            bcore = core.float() if use_accum32 else core
            res = torch.tensordot(a, bcore, dims=([r_prev_idx, i_idx], [0, 2]))
            # 结果形状：y 未收缩轴（按原顺序） + 核未收缩轴 [o_k, r_k]
            # 更新 i_axes：移除被收缩的轴，并将其后面的轴索引减 1（由于去掉了一个轴）
            i_axes = [ax for ax in i_axes if ax != i_idx]
            i_axes = [ax - 1 if ax > i_idx else ax for ax in i_axes]
            # 新增轴位置：o_k 位于末尾-1，r_k 位于末尾
            r_prev_idx = res.ndim - 1
            y = res

        # 去掉最后的 r_d=1 轴
        y = y.squeeze(-1)

        # 直接展平输出轴为 out_f（o 轴已按顺序位于批维之后）
        y = y.contiguous().view(*B_dims, int(self.out_f))

        if guard_on:
            y = self._clean_finite(y)
        tgt = x.dtype
        if guard_on and tgt in (torch.float16, torch.bfloat16):
            y = torch.clamp(y, min=-safe_clamp, max=safe_clamp)
        return y.to(tgt)

    def _forward_mpo_cotengra(
        self, x: torch.Tensor, B_dims: tuple, in_factors: list, out_factors: list, use_accum32: bool
    ) -> torch.Tensor:
        import os

        # 环境与后端
        try:
            import opt_einsum as oe
        except Exception as e:
            raise RuntimeError("opt_einsum is required for CoTengra backend") from e
        try:
            import cotengra as ctg
        except Exception as e:
            raise RuntimeError("cotengra is required for CoTengra backend") from e

        # 清洗输入
        guard_on = os.getenv("MPO_STREAM_GUARD", "0") == "1"
        safe_clamp = float(os.getenv("MPO_STREAM_SAFE_CLAMP", str(self.default_stream_safe_clamp)))
        if guard_on:
            x = self._clean_finite(x)

        device = x.device
        accum_dt = torch.float32 if use_accum32 else x.dtype

        # 预处理形状
        bdim = 1
        for v in B_dims:
            bdim *= int(v)
        d = len(in_factors)
        # 固定 batch 维以稳定 ctg 计划（可选）
        try:
            fixed_bdim = int(os.getenv("MPO_FIXED_BDIM", "0"))
        except Exception:
            fixed_bdim = 0
        use_fixed_b = fixed_bdim > 0 and int(bdim) <= int(fixed_bdim)
        bdim_key = int(fixed_bdim) if use_fixed_b else int(bdim)

        # 构建 einsum 式：  b i1 i2 ... id, r0 o1 i1 r1, r1 o2 i2 r2, ... -> b o1 o2 ... od
        get_sym = oe.get_symbol
        b_sym = get_sym(0)
        i_syms = [get_sym(1 + k) for k in range(d)]
        o_syms = [get_sym(1 + d + k) for k in range(d)]
        r_syms = [get_sym(1 + 2 * d + k) for k in range(d + 1)]
        x_sub = b_sym + "".join(i_syms)
        core_subs = [r_syms[k] + o_syms[k] + i_syms[k] + r_syms[k + 1] for k in range(d)]
        out_sub = b_sym + "".join(o_syms)
        equation = ",".join([x_sub] + core_subs) + "->" + out_sub

        # 组装形状
        shapes = [(int(bdim_key), *[int(v) for v in in_factors])]
        core_shapes = []
        for c in self.cores:
            shp = tuple(int(s) for s in c.shape)
            core_shapes.append(shp)
            shapes.append(shp)

        # 优化器与表达式缓存
        max_time = float(os.getenv("MPO_CTG_MAXTIME", "0.2"))
        max_repeats = int(os.getenv("MPO_CTG_REPEATS", "16"))
        minimize = os.getenv("MPO_CTG_TARGET", "flops").strip().lower()
        # 并行优化：True=自动多核，False=单线程，int=指定进程数
        parallel_opt = os.getenv("MPO_CTG_PARALLEL", "auto").strip().lower()
        if parallel_opt in ("true", "1", "auto"):
            parallel = True  # 自动使用所有CPU核心
        elif parallel_opt in ("false", "0"):
            parallel = False
        else:
            try:
                parallel = int(parallel_opt)  # 指定进程数
            except:
                parallel = True
        try:
            optimizer = ctg.HyperOptimizer(
                max_time=max_time,
                max_repeats=max_repeats,
                minimize=minimize if minimize in ("flops", "size") else "flops",
                parallel=parallel,
                progbar=False,
            )
        except Exception:
            optimizer = "greedy"

        cache_key = (
            int(bdim_key),
            tuple(int(v) for v in in_factors),
            tuple(int(v) for v in out_factors),
            tuple(core_shapes),
        )
        if not hasattr(self, "_ctg_expr_cache"):
            self._ctg_expr_cache = {}
        expr = self._ctg_expr_cache.get(cache_key)
        if expr is None:
            expr = oe.contract_expression(equation, *shapes, optimize=optimizer)
            self._ctg_expr_cache[cache_key] = expr

        # 准备操作数（按累加精度）
        x_acc = x.reshape(int(bdim), int(self.in_f)).view(int(bdim), *[int(v) for v in in_factors])
        x_acc = x_acc.float() if use_accum32 else x_acc
        if use_fixed_b and int(bdim) < int(fixed_bdim):
            pad_rows = int(fixed_bdim) - int(bdim)
            pad_shape = (pad_rows, *x_acc.shape[1:])
            x_pad = torch.zeros(pad_shape, dtype=x_acc.dtype, device=x_acc.device)
            x_acc = torch.cat([x_acc, x_pad], dim=0)
        cores = self._prepare_stream_cores(device, torch.float32 if use_accum32 else x.dtype)

        # 收缩
        y_acc = expr(x_acc, *cores, backend="torch")  # (B_key, *o_factors)
        if use_fixed_b and int(bdim) < int(fixed_bdim):
            y_acc = y_acc[: int(bdim), ...]
        y_acc = y_acc.contiguous().view(*B_dims, int(self.out_f))

        # 结束清洗/钳位与返回 dtype
        if guard_on:
            y_acc = self._clean_finite(y_acc)
            if x.dtype in (torch.float16, torch.bfloat16):
                y_acc = torch.clamp(y_acc, min=-float(safe_clamp), max=float(safe_clamp))
        return y_acc.to(x.dtype)

    def _forward_mpo_cuquantum(
        self, x: torch.Tensor, B_dims: tuple, in_factors: list, out_factors: list, use_accum32: bool
    ) -> torch.Tensor:
        """
        使用 cuQuantum tensornet 执行 MPO 收缩
        联合方案: cotengra 找路径 + cuQuantum 执行
        """
        import os

        try:
            from cuquantum import tensornet as tn
        except ImportError as e:
            raise RuntimeError(f"cuquantum.tensornet import failed: {e}. Please install: pip install cuquantum") from e
        except Exception as e:
            raise RuntimeError(f"cuquantum.tensornet unexpected error: {e}") from e

        # 清洗输入
        guard_on = os.getenv("MPO_STREAM_GUARD", "0") == "1"
        safe_clamp = float(os.getenv("MPO_STREAM_SAFE_CLAMP", str(self.default_stream_safe_clamp)))
        if guard_on:
            x = self._clean_finite(x)

        device = x.device
        accum_dt = torch.float32 if use_accum32 else x.dtype

        # 预处理形状
        bdim = 1
        for v in B_dims:
            bdim *= int(v)
        d = len(in_factors)

        # 准备操作数
        x_acc = x.reshape(int(bdim), int(self.in_f)).view(int(bdim), *[int(v) for v in in_factors])
        x_acc = x_acc.float() if use_accum32 else x_acc
        cores = self._prepare_stream_cores(device, torch.float32 if use_accum32 else x.dtype)

        # 构建 einsum 式: b i1 i2 ... id, r0 o1 i1 r1, r1 o2 i2 r2, ... -> b r0 o1 o2 ... od rd
        # 对于 3 核 MPO (最常见情况)
        if d == 3:
            # x: [b, i1, i2, i3]
            # core0: [r0, o1, i1, r1]
            # core1: [r1, o2, i2, r2]
            # core2: [r2, o3, i3, r3]
            # 输出: [b, r0, o1, o2, o3, r3]
            equation = "bijk,roip,pqjr,rstk->boqst"

            # 使用 cuQuantum tensornet 执行收缩
            y_acc = tn.contract(equation, x_acc, cores[0], cores[1], cores[2])

            # 去掉 r0 和 r3 维度 (通常为 1)
            y_acc = y_acc.squeeze(1).squeeze(-1)  # [b, o1, o2, o3]
        else:
            # 通用情况：构建完整的 einsum 表达式
            try:
                import opt_einsum as oe
            except Exception as e:
                raise RuntimeError("opt_einsum is required for general MPO") from e

            get_sym = oe.get_symbol
            b_sym = get_sym(0)
            i_syms = [get_sym(1 + k) for k in range(d)]
            o_syms = [get_sym(1 + d + k) for k in range(d)]
            r_syms = [get_sym(1 + 2 * d + k) for k in range(d + 1)]
            x_sub = b_sym + "".join(i_syms)
            core_subs = [r_syms[k] + o_syms[k] + i_syms[k] + r_syms[k + 1] for k in range(d)]
            out_sub = b_sym + r_syms[0] + "".join(o_syms) + r_syms[d]
            equation = ",".join([x_sub] + core_subs) + "->" + out_sub

            # 使用 cuQuantum tensornet 执行收缩
            y_acc = tn.contract(equation, x_acc, *cores)

            # 去掉 r0 和 rd 维度
            y_acc = y_acc.squeeze(1).squeeze(-1)

        # 重塑输出
        y_acc = y_acc.contiguous().view(*B_dims, int(self.out_f))

        # 结束清洗/钳位与返回 dtype
        if guard_on:
            y_acc = self._clean_finite(y_acc)
            if x.dtype in (torch.float16, torch.bfloat16):
                y_acc = torch.clamp(y_acc, min=-float(safe_clamp), max=float(safe_clamp))
        return y_acc.to(x.dtype)

    # ---------- dense 路径（训练推荐；支持 chunk+ckpt） ----------
    @staticmethod
    def _checkpointable_matmul(x_chunk: torch.Tensor, w_chunk: torch.Tensor) -> torch.Tensor:
        return x_chunk @ w_chunk.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        路径选择策略（可由环境变量覆盖）：
        - 默认：eval → STREAM(cotengra)；train → STREAM(cotengra)
        - 覆盖：
            * MPO_EVAL_PATH  ∈ {'auto','dense','stream'}，默认 'auto'（即 stream）
            * MPO_TRAIN_PATH ∈ {'auto','dense','stream'}，默认 'auto'（即 stream）
        - 若选择 'stream' 但不满足 4D 条件或执行异常：
            * MPO_STRICT_STREAM='1' → 直接报错
            * 否则回退到 dense
        其他：
        - dense 路径支持分块 & checkpoint（受 MPO_MATMUL_CHUNK / MPO_CHECKPOINT 控制）
        - eval+dense 支持 cache dense 权重（cache_on_eval=True 时）
        """
        # ==========================================
        # [新增核心逻辑]：推理时的输入逆变换
        # ==========================================
        if getattr(self, "scale_inv", None) is not None:
            # 输入 x 形状通常是 [..., in_f]
            # 这里利用广播机制逐元素相乘，极度省显存且速度极快
            x = x * self.scale_inv.to(x.dtype)
            
        import os

        layer = getattr(self, "layer_name", "(unknown)")
        debug = os.getenv("MPO_DEBUG", "0") == "1"

        is_train = self.training
        # 读取 phase 定制开关
        phase_key = "MPO_TRAIN_PATH" if is_train else "MPO_EVAL_PATH"
        override = os.getenv(phase_key, "auto").strip().lower()
        # 兼容写法：允许 d/s 缩写
        if override in ("d", "s"):
            override = {"d": "dense", "s": "stream"}[override]
        if override not in ("auto", "dense", "stream"):
            override = "auto"

        # 缺省路径：eval→stream；train→stream（默认都使用 cotengra 优化）
        target_path = "stream" if override == "auto" else override
        strict_stream = os.getenv("MPO_STRICT_STREAM", "0") == "1"

        if debug:
            print(
                f"[MPOLinear:{layer}] phase={'train' if is_train else 'eval'} "
                f"override={override} -> target={target_path}"
            )

        # ---------- STREAM 路径（按后端选择 ctg 或 classic） ----------
        if target_path == "stream":
            # 后端：默认 ctg，可通过 MPO_CLASSIC_BACKEND=classic 切换为纯 tensordot
            try:
                backend = os.getenv("MPO_CLASSIC_BACKEND", "cotengra").strip().lower()
            except Exception:
                backend = "cotengra"

            # 环境读取（与 classic 对齐）
            def _get_env_bool(key: str, default: bool) -> bool:
                val = os.getenv(key, None)
                if val is None:
                    return bool(default)
                v = val.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return True
                if v in ("0", "false", "no", "n", "off"):
                    return False
                try:
                    return bool(int(v))
                except Exception:
                    return bool(default)

            use_accum32 = _get_env_bool("MPO_STREAM_ACCUM32", self.default_stream_accum32)
            strict_stream = os.getenv("MPO_STRICT_STREAM", "0") == "1"

            # 形状准备与一致性校验
            if not getattr(self, "_stream_ready", False):
                ok = self._init_stream_factors()
                if not ok:
                    if strict_stream:
                        raise RuntimeError(
                            "[MPOLinear] strict stream requires 4D MPO cores; fallback to dense is disabled"
                        )
                    # 回退到 dense
                    
                    #W32 = self._build_full_weight_fp32()
                    #x32 = x.to(torch.float32)
                    #W32 = W32.to(x32.device)
                    #y32 = x32 @ W32.t()
                    #return y32.to(x.dtype)
                    # ---------------------------------------------------------
                    # 替换原先的 5 行代码
                    # ---------------------------------------------------------
                    # ---------------------------------------------------------
                    # 极其干净的前向传播
                    # ---------------------------------------------------------
                    # 1. 这一步会自动去调用 ring_ops.py，安全组装出 16MB 的 Dense 矩阵
                    W32 = self._build_full_weight_fp32() 
                    
                    # 2. 对齐精度和设备
                    x32 = x.to(torch.float32)
                    W32 = W32.to(x32.device)
                    
                    # 3. 最普通的矩阵乘法，避开所有高维爆炸
                    y32 = x32 @ W32.t()
                    
                    return y32.to(x.dtype)
                    # ---------------------------------------------------------
                        
                    return y32.to(x.dtype)
                    # ---------------------------------------------------------

            B_dims = x.shape[:-1]
            in_factors = list(self._in_factors)
            out_factors = list(self._out_factors)

            if backend in ("cotengra", "ctg"):
                if debug:
                    print(f"[MPOLinear:{layer}] path=STREAM backend=ctg")
                return self._forward_mpo_cotengra(x, B_dims, in_factors, out_factors, use_accum32)
            elif backend in ("cuquantum", "cuq"):
                if debug:
                    print(f"[MPOLinear:{layer}] path=STREAM backend=cuquantum")
                return self._forward_mpo_cuquantum(x, B_dims, in_factors, out_factors, use_accum32)
            else:
                if debug:
                    print(f"[MPOLinear:{layer}] path=STREAM backend=classic")
                return self._forward_mpo_classic(x)

        # ---------- CLASSIC MPO 收缩（与 dense 并列存在） ----------
        # 这里不再由环境变量控制；stream 已改为默认 classic 实现。

        # ---------- DENSE 路径 ----------
        if debug:
            reason = []
            if target_path != "stream":
                reason.append(f"target={target_path}")
            print(f"[MPOLinear:{layer}] path=DENSE ({'|'.join(reason) if reason else 'fallback'})")

        # 计算精度控制：默认使用输入 dtype（fp16/bf16）；当 MPO_DENSE_FP32=1 时强制 fp32
        from torch.amp import autocast

        dense_safe_clamp = float(self.default_dense_safe_clamp)
        dense_input_clamp = float(self.default_dense_input_clamp)
        use_fp32_dense = os.environ.get("MPO_DENSE_FP32", "0") == "1"
        compute_dtype = torch.float32 if use_fp32_dense else x.dtype
        with autocast(device_type="cuda", enabled=False):
            x = self._clean_finite(x)
            if dense_input_clamp > 0:
                x = torch.clamp(x, min=-dense_input_clamp, max=dense_input_clamp)
            try:
                log_tensor(f"{layer}.MPOLinear.in", x, raise_on_bad=False)  # type: ignore
            except Exception:
                pass

            # 生成/缓存 dense 权重（先构建 fp32，再按 compute_dtype 缓存/使用）
            if self.training or (not self.cache_on_eval):
                W_base = self._build_full_weight_fp32()
                W_comp = W_base if use_fp32_dense else W_base.to(compute_dtype)
                if debug:
                    print(
                        f"[MPOLinear:{layer}] build_full_weight (no cache), W={tuple(W_comp.shape)} dtype={W_comp.dtype}"
                    )
            else:
                W_cached = getattr(self, "weight_full", None)
                if W_cached is None:
                    with torch.no_grad():
                        W_base = self._build_full_weight_fp32()
                        W_comp = W_base if use_fp32_dense else W_base.to(compute_dtype)
                        self.register_buffer("weight_full", W_comp, persistent=False)
                    if debug:
                        print(
                            f"[MPOLinear:{layer}] build_full_weight -> cached, W={tuple(W_comp.shape)} dtype={W_comp.dtype}"
                        )
                else:
                    W_comp = W_cached
                    if debug:
                        print(
                            f"[MPOLinear:{layer}] reuse cached dense weight, W={tuple(W_comp.shape)} dtype={W_comp.dtype}"
                        )

            x_comp = x.to(compute_dtype)
            x2d = x_comp.view(-1, x_comp.shape[-1])
            out_rows = int(W_comp.shape[0])

            row_chunk = int(os.environ.get("MPO_MATMUL_CHUNK", "4096"))
            use_ckpt = os.environ.get("MPO_CHECKPOINT", "1") == "1"

            if debug:
                print(
                    f"[MPOLinear:{layer}] matmul row_chunk={row_chunk}, "
                    f"use_ckpt={self.training and use_ckpt} compute_dtype={x2d.dtype}"
                )

            y_chunks: List[torch.Tensor] = []
            for s in range(0, out_rows, row_chunk):
                e = min(out_rows, s + row_chunk)
                Wblk = W_comp[s:e, :].to(x2d.device, dtype=x2d.dtype, non_blocking=True)
                if self.training and use_ckpt:
                    y_blk = checkpoint(self._checkpointable_matmul, x2d, Wblk, use_reentrant=False)
                else:
                    y_blk = self._checkpointable_matmul(x2d, Wblk)
                y_chunks.append(y_blk)
                del Wblk, y_blk

            y2d = torch.cat(y_chunks, dim=1) if len(y_chunks) > 1 else y_chunks[0]
            y32 = y2d.view(*x_comp.shape[:-1], out_rows)

            y32 = self._clean_finite(y32)
            tgt = x.dtype
            if tgt in (torch.float16, torch.bfloat16):
                # 先用可控安全阈值，再用 dtype 极限兜底
                y32 = torch.clamp(y32, min=-dense_safe_clamp, max=dense_safe_clamp)
                finfo = torch.finfo(tgt)
                y32 = torch.clamp(y32, min=finfo.min, max=finfo.max)
            y = y32.to(tgt)

        if debug:
            print(f"[MPOLinear:{layer}] dense-forward OK, out={tuple(y.shape)} dtype={y.dtype}")
        return self._clean_finite(y)


# ────────────────────────────────────────────────────────────────
# ========================= make_mpo_from_config（无误差参数） =========================


# ============================================================
# 导出
# ============================================================

__all__ = [
    "MPOLinear",
]


# ============================================================
# 使用说明
# ============================================================

"""
使用示例:

    from mpo_modules.core import MPOLinear
    import torch
    
    # 创建 MPO 层
    cores = [...]  # 你的 MPO 核
    mpo = MPOLinear(in_f=4096, out_f=4096, cores=cores)
    
    # 前向传播
    x = torch.randn(2, 4096)
    y = mpo(x)
    
环境变量控制:
    
    - MPO_EVAL_PATH / MPO_TRAIN_PATH: 'auto', 'dense', 'stream'
    - MPO_CLASSIC_BACKEND: 'cotengra', 'classic'
    - MPO_STREAM_ACCUM32: '0' / '1'
    - MPO_STREAM_SAFE_CLAMP: float
    - MPO_DEBUG: '0' / '1'
    
详见类文档字符串。
"""
