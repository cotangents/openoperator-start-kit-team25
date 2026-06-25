import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor cond, torch::Tensor gamma_weight, torch::Tensor gamma_bias, torch::Tensor beta_weight, torch::Tensor beta_bias, int dim, int cond_dim);
"""

bang_func_ext = load_inline(
    name="Conditional_LayerNorm",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.gamma_proj = nn.Linear(cond_dim, dim)
        self.beta_proj = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(
            x, cond,
            self.gamma_proj.weight, self.gamma_proj.bias,
            self.beta_proj.weight, self.beta_proj.bias,
            self.dim, self.cond_dim
        )