import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, int num_features);
"""

bang_func_ext = load_inline(
    name="Batch_norm_1D",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(
            x,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.num_features
        )