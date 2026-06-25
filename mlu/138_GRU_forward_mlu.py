import torch
import torch.nn as nn
import torch_mlu
from torch_mlu.utils.cpp_extension import load_inline

bang_func_source = """
"""

bang_func_cpp_src = """
torch::Tensor bang_func(
    torch::Tensor x,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    int input_size,
    int hidden_size,
    int num_layers
);
"""

bang_func_ext = load_inline(
    name="GRU_forward",
    cpp_sources=bang_func_cpp_src,
    bang_sources=bang_func_source,
    functions=["bang_func"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcnrt"],
    extra_bang_cflags=['-O3', '-lm'],
)

class ModelNew(nn.Module):
    def __init__(
        self,
        weight_ih_l0: torch.Tensor,
        weight_hh_l0: torch.Tensor,
        bias_ih_l0: torch.Tensor,
        bias_hh_l0: torch.Tensor,
        weight_ih_l1: torch.Tensor,
        weight_hh_l1: torch.Tensor,
        bias_ih_l1: torch.Tensor,
        bias_hh_l1: torch.Tensor,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.weight_ih_l0 = weight_ih_l0
        self.weight_hh_l0 = weight_hh_l0
        self.bias_ih_l0 = bias_ih_l0
        self.bias_hh_l0 = bias_hh_l0
        self.weight_ih_l1 = weight_ih_l1
        self.weight_hh_l1 = weight_hh_l1
        self.bias_ih_l1 = bias_ih_l1
        self.bias_hh_l1 = bias_hh_l1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bang_func_ext.bang_func(
            x,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            self.weight_ih_l1,
            self.weight_hh_l1,
            self.bias_ih_l1,
            self.bias_hh_l1,
            self.input_size,
            self.hidden_size,
            self.num_layers,
        )
