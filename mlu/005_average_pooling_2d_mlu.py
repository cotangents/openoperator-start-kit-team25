import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x,int kernel_size);
"""

bang_func_ext = load_inline(
    name="average_pooling_2d",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self,kernel_size:int):
        super().__init__()
        self.kernel_size=kernel_size

    def forward(self, x):
        return bang_func_ext.bang_func(x,self.kernel_size)
