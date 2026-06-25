import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, int size, double alpha, double beta, double k);
"""

bang_func_ext = load_inline(
    name="LocalResponseNorm",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(x, self.size, self.alpha, self.beta, self.k)