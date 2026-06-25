import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor depthwise_kernel, torch::Tensor pointwise_kernel, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation);
"""

bang_func_ext = load_inline(
    name="conv_depthwise_separable_2D_bang",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, depthwise_kernel: torch.Tensor, pointwise_kernel: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise_kernel = depthwise_kernel
        self.pointwise_kernel = pointwise_kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(
            x,
            self.depthwise_kernel,
            self.pointwise_kernel,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
