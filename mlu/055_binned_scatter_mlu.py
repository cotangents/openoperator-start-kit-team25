import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor indices, torch::Tensor weights, torch::Tensor bins, int top_k);
"""

bang_func_ext = load_inline(
    name="binned_scatter",
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

    def forward(self, x, indices, weights, bins, top_k):
        return bang_func_ext.bang_func(x, indices, weights, bins, top_k)
