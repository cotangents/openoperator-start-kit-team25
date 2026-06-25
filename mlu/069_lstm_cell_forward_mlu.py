import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor input, torch::Tensor hidden, torch::Tensor bias1, torch::Tensor bias2, torch::Tensor cx, int hsz, int totalElements);
"""

bang_func_ext = load_inline(
    name="lstm_cell_forward",
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

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor, cx: torch.Tensor, hsz: int, totalElements: int) -> torch.Tensor:
        return bang_func_ext.bang_func(input, hidden, bias1, bias2, cx, hsz, totalElements)