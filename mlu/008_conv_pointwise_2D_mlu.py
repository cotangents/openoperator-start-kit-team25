import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor kernel, int in_channels, int out_channels, bool bias);
"""

bang_func_ext = load_inline(
    name="conv_pointwise_2D",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, kernel: torch.Tensor, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(x, self.kernel, self.in_channels, self.out_channels, self.bias)
