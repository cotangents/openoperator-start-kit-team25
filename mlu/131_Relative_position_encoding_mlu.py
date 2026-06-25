import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(int seq_len, int max_seq, int d_model);
"""

bang_func_ext = load_inline(
    name="Relative_position_encoding",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(self, max_seq: int, d_model: int):
        super().__init__()
        self.max_seq = max_seq
        self.d_model = d_model

    def forward(self, seq_len: int) -> torch.Tensor:
        return bang_func_ext.bang_func(seq_len, self.max_seq, self.d_model)