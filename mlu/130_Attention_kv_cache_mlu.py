import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_new, torch::Tensor v_new);
"""

bang_func_ext = load_inline(
    name="Attention_kv_cache",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                k_new: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(q, k_cache, v_cache, k_new, v_new)