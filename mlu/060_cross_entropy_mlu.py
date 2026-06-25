import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor _input, torch::Tensor target, torch::Tensor label_smoothing, torch::Tensor reduce_loss, torch::Tensor dist_process_group, torch::Tensor ignore_idx);
"""

bang_func_ext = load_inline(
    name="cross_entropy",
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

    def forward(self, _input, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx):
        return bang_func_ext.bang_func(_input, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx)