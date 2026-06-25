import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor input_log_prob, torch::Tensor target_prob);
"""

bang_func_ext = load_inline(
    name="KL_Divergence_Loss",
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

    def forward(self, input_log_prob: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(input_log_prob, target_prob)