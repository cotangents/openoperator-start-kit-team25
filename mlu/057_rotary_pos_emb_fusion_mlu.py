import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
std::vector<torch::Tensor> bang_func(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin, int unsqueeze_dim);
"""

bang_func_ext = load_inline(
    name="rotary_pos_emb_fusion",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, unsqueeze_dim: int = 1):
        super().__init__()
        self.unsqueeze_dim = unsqueeze_dim

    def forward(self, q, k, cos, sin):
        return bang_func_ext.bang_func(q, k, cos, sin, self.unsqueeze_dim)
