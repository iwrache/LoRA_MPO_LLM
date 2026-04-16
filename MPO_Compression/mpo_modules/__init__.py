"""
mpo_modules — MPO (Matrix Product Operator) compression for LLMs.

Core components:
- MPOLinear: drop-in replacement for nn.Linear using TT/MPO decomposition
- factor_linear_mpo: decompose a dense linear layer into MPO form
- load_mpo_model / replace_llama_linears: model-level compression utilities
"""

from .core import MPOLinear
from .factorization import (
    estimate_mpo_bond_dim,
    factor_linear_mpo,
    get_mpo_compression_ratio,
    robust_svd_split,
)
from .helpers import find_factors_balanced, log_tensor
from .patches import (
    apply_all_safety_patches,
    apply_lm_head_input_fp32,
    apply_rmsnorm_fp32,
    apply_sdpa_safety_patches,
    install_residual_guards,
    unpatch_llama_attention,
)
from .ring_ops import chain_to_ring, contract_ring, get_ring_rank_info, matrix_ring_svd
from .tt_ops import get_tt_rank_info, matrix_tt_svd, mpo_right_apply_operator, tt_round_4d_cores

# model_utils has heavy deps (transformers, accelerate, safetensors);
# import lazily so lightweight usage (e.g. just MPOLinear) doesn't fail.
_MODEL_UTILS_AVAILABLE = False
try:
    from .model_utils import (
        check_tied,
        convert_mpo_to_dense,
        load_mpo_model,
        make_mpo_from_config,
        replace_llama_linears,
        replace_llama_linears_by_cfg,
        replace_llama_linears_by_maps,
        retie_lm_head,
    )

    _MODEL_UTILS_AVAILABLE = True
except ImportError:
    pass

__version__ = "0.1.0"

__all__ = [
    "MPOLinear",
    "factor_linear_mpo",
    "get_mpo_compression_ratio",
    "estimate_mpo_bond_dim",
    "robust_svd_split",
    "find_factors_balanced",
    "log_tensor",
    "matrix_tt_svd",
    "mpo_right_apply_operator",
    "tt_round_4d_cores",
    "get_tt_rank_info",
    "apply_sdpa_safety_patches",
    "apply_rmsnorm_fp32",
    "apply_lm_head_input_fp32",
    "install_residual_guards",
    "unpatch_llama_attention",
    "apply_all_safety_patches",
    "matrix_ring_svd",
    "contract_ring",
    "chain_to_ring",
    "get_ring_rank_info",
]

if _MODEL_UTILS_AVAILABLE:
    __all__ += [
        "make_mpo_from_config",
        "load_mpo_model",
        "convert_mpo_to_dense",
        "replace_llama_linears_by_cfg",
        "replace_llama_linears_by_maps",
        "replace_llama_linears",
        "retie_lm_head",
        "check_tied",
    ]
