import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor kernel, int in_channels, int out_channels, int kernel_size, int dilation, int padding);
"""

bang_func_ext = load_inline(
    name="Dilated_conv_2D",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, kernel: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 2, padding: int = 2):
        super().__init__()
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(x, self.kernel, self.in_channels, self.out_channels, self.kernel_size, self.dilation, self.padding)
