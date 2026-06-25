import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor attn_weight, torch::Tensor mask, double scale);
"""

bang_func_ext = load_inline(
    name="Scaled_masked_softmax",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, scale: float = None):
        super().__init__()
        self.scale = scale

    def forward(self, attn_weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(attn_weight, mask, self.scale if self.scale is not None else 1.0)