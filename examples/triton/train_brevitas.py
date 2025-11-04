from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant_tensor import QuantTensor
from torch.library import register_autograd
import triton.language as tl
import brevitas.nn as qnn
import triton
import torch
import os


# Kernel implementation of matrix-vector multiplication
@triton.jit
def _mvm_kernel(
    A_ptr,
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M], dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_idx = (offs_m[:, None] * K) + offs_k[None, :]
        a = tl.load(A_ptr + a_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)
    tl.store(y_ptr + offs_m, acc, mask=mask_m)


# Kernel wrapper function: Checks inputs and launches the kernel
def triton_mvm(A: torch.Tensor,
               x: torch.Tensor,
               block_m: int = 128,
               block_k: int = 128) -> torch.Tensor:
    """
    A: [M, K], x: [K]  (beide CUDA, contiguous)
    returns y: [M]
    """
    assert A.is_cuda and x.is_cuda
    assert A.dim() == 2 and x.dim() == 1
    M, K = A.shape
    assert x.numel() == K
    assert A.dtype == torch.int32 and x.dtype == torch.int32

    A_c = A.contiguous()
    x_c = x.contiguous()
    y = torch.empty(M, device=A.device, dtype=torch.int32)

    grid = (triton.cdiv(M, block_m), )
    _mvm_kernel[grid](
        A_c,
        x_c,
        y,
        M,
        K,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return y


# Create a namespace 'mylib' and register the mvm operator
lib = torch.library.Library("mylib", "DEF")
lib.define("mvm(Tensor A, Tensor x) -> Tensor")


# Register kernel implementation for CUDA and 'mvm' operator
@torch.library.impl("mylib::mvm", "CUDA")
def _mvm_impl_cuda(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return triton_mvm(A, x)


# Register abstract (shape and dtype inference) for 'mvm' operator
@torch.library.register_fake("mylib::mvm")
def _mvm_abstract(A: torch.Tensor, x: torch.Tensor):
    M, K = A.shape
    assert x.shape == (K, )
    return A.new_empty((M, ), dtype=torch.int32)


# Setup function for autograd: saves A and x for backward pass
def _mvm_setup_context(ctx, inputs, output):
    A, x = inputs
    ctx.save_for_backward(A, x)


# Backward function for autograd: computes gradients w.r.t. A and x
# Function: y = A @ x
# δL(y)/δA = δL(y)/δy * δy(A)/δA = grad_y * x
# δL(y)/δx = δL(y)/δy * δy(x)/δx = grad_y * A
# dA.shape = A.shape, dx.shape = x.shape
def _mvm_backward(ctx, grad_y):
    A, x = ctx.saved_tensors
    dA = grad_y.unsqueeze(1) * x.unsqueeze(0)
    dx = A.t().matmul(grad_y)
    return dA, dx


# Link autograd functions to the 'mvm' operator
register_autograd("mylib::mvm", _mvm_backward, setup_context=_mvm_setup_context)


class TinyMLP(torch.nn.Module):

    def __init__(self, in_features=128, hidden=64, out_features=10):
        super().__init__()
        self.fc1 = qnn.QuantLinear(in_features,
                                   hidden,
                                   bias=True,
                                   input_quant=Int8ActPerTensorFloat)
        self.act = torch.nn.ReLU()
        self.fc2 = qnn.QuantLinear(hidden,
                                   out_features,
                                   bias=True,
                                   input_quant=Int8ActPerTensorFloat)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# Test standalone Triton implementation
def check():
    M, K = 513, 777
    A = torch.randint(low=-10, high=10, size=(M, K), device="cuda", dtype=torch.int32)
    x = torch.randint(low=-10, high=10, size=(K, ), device="cuda", dtype=torch.int32)
    y_ref = (A.to(torch.int64) * x.to(torch.int64)).sum(dim=1).to(torch.int32)
    y_triton = torch.ops.mylib.mvm(A, x)
    max_err = (y_ref - y_triton).abs().max().item()
    print("Triton MVM vs. torch mvm: max abs error:", max_err)


check()

# Create input for TinyMLP and run a forward pass
B, K = 16, 128
model = TinyMLP()
x = torch.randn(B, K, device="cpu", dtype=torch.float32)
y_ref_model = model(x)


class QuantLinearIntTriton(qnn.QuantLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        quant_input = self.input_quant(x)
        quant_weight = self.quant_weight(quant_input)

        int_input = (quant_input / quant_input.scale).to(torch.int32)
        int_weight = (quant_weight / quant_weight.scale).to(torch.int32)
        if quant_input.dim() == 2:    # [B, K]
            y_val = torch.stack([
                torch.ops.mylib.mvm(int_weight, int_input[b]) for b in range(quant_input.shape[0])
            ],
                                dim=0)
        else:    # [K]
            y_val = torch.ops.mylib.mvm(int_weight, int_input)

        y_val = y_val.to(torch.float32)
        y_val = y_val * quant_input.scale * quant_weight.scale

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
            y_val = y_val + quant_bias

        quant_output = self.output_quant(y_val)
        return quant_output


# Replace fc1 and f2 forward with custom forward using triton_mvm
def replace_brevitas_linear_with_int_triton(module: torch.nn.Module,
                                            return_quanttensor: bool = False):
    for name, child in list(module.named_children()):
        if isinstance(child, qnn.QuantLinear):
            new = QuantLinearIntTriton(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
                weight_quant=getattr(child, "weight_quant", None),
                bias_quant=getattr(child, "bias_quant", None),
                input_quant=getattr(child, "input_quant", None),
                output_quant=getattr(child, "output_quant", None),
                return_quant_tensor=getattr(child, "return_quant_tensor", False),
            )
            new.load_state_dict(child.state_dict(), strict=False)
            new.to(device=next(child.parameters()).device, dtype=next(child.parameters()).dtype)
            new.train(child.training)
            setattr(module, name, new)
        else:
            replace_brevitas_linear_with_int_triton(child, return_quanttensor=return_quanttensor)


replace_brevitas_linear_with_int_triton(model)
model = model.cuda()
x = x.cuda()

if os.getenv("TRITON_INTERPRET") == "1":
    y = model(x)
else:
    opt_model = torch.compile(model, backend="inductor")
    y = opt_model(x)

# Compare outputs
y = y.cpu()
max_err = (y_ref_model - y).abs().max().item()
print("TinyMLP with replaced linear layers vs. torch: max abs error:", max_err)
