import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func({forward_parameter});
"""

bang_func_ext = load_inline(
    name="{op_name}",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
)

class ModelNew(nn.Module):
    def __init__(self,{init_parameter}):
        super().__init__()
        {self.xxx = xxx}

    def forward(self,{forward_parameter}):
        return bang_func_ext.bang_func({forward_parameter})
