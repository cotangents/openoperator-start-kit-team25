import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(torch::Tensor q, torch::Tensor k, torch::Tensor v, double alpha, torch::Tensor seq_offsets, int max_seq_len, torch::Tensor context_size, torch::Tensor history_size, torch::Tensor realtime_size, torch::Tensor history_causal, torch::Tensor context_expand, torch::Tensor label_lens, int preference);
"""

bang_func_ext = load_inline(
    name="hstu_mask_var_len_fwd",
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

    def forward(self, q, k, v, alpha, seq_offsets, max_seq_len, context_size, history_size, realtime_size, history_causal, context_expand, label_lens, preference):
        return bang_func_ext.bang_func(q, k, v, alpha, seq_offsets, max_seq_len, context_size, history_size, realtime_size, history_causal, context_expand, label_lens, preference)
