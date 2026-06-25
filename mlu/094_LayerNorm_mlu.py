import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int normalized_shape);
"""

bang_func_ext = load_inline(
    name="LayerNorm",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(x, self.weight, self.bias, self.normalized_shape)