import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor query, torch::Tensor key_value, int d_model, int num_heads);
"""

bang_func_ext = load_inline(
    name="Cross_Attention",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(query, key_value, self.d_model, self.num_heads)