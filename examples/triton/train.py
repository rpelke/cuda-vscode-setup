import torch
import triton
import triton.language as tl


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

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_idx = (offs_m[:, None] * K) + offs_k[None, :]
        a = tl.load(A_ptr + a_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)
    tl.store(y_ptr + offs_m, acc, mask=mask_m)


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

    A_c = A.contiguous()
    x_c = x.contiguous()
    y = torch.empty(M, device=A.device, dtype=torch.float32)

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


lib = torch.library.Library("cim", "DEF")
lib.define("mvm(Tensor A, Tensor x) -> Tensor")


@torch.library.impl("cim::mvm", "CUDA")
def _mvm_impl_cuda(A: torch.Tensor, x: torch.Tensor):
    return triton_mvm(A, x)


@torch.library.register_fake("cim::mvm")
def _mvm_abstract(A: torch.Tensor, x: torch.Tensor):
    M, K = A.shape
    assert x.shape == (K, )
    return A.new_empty((M, ), dtype=torch.float32)


from torch.library import register_autograd


def _mvm_setup_context(ctx, inputs, output):
    A, x = inputs
    ctx.save_for_backward(A, x)


def _mvm_backward(ctx, grad_y):
    A, x = ctx.saved_tensors
    dA = grad_y.unsqueeze(1) * x.unsqueeze(0)
    dx = A.t().matmul(grad_y)
    return dA, dx


register_autograd("cim::mvm", _mvm_backward, setup_context=_mvm_setup_context)


class TinyMLP(torch.nn.Module):

    def __init__(self, in_features=784, hidden=256, out_features=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


# Test standalone mvm operation
def check():
    M, K = 513, 777
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    x = torch.randn(K, device="cuda", dtype=torch.float32)
    y_ref = A @ x
    y_triton = torch.ops.cim.mvm(A, x)
    max_err = (y_ref - y_triton).abs().max().item()
    print("max abs error:", max_err)


check()

# Create input for TinyMLP and run a forward pass
B, K = 16, 784
model = TinyMLP()
x = torch.randn(B, K, device="cpu", dtype=torch.float32)
y_ref_model = model(x)


class CustomLinear(torch.nn.Linear):

    def forward(self, x):
        ys = [torch.ops.cim.mvm(self.weight, x[b]) for b in range(x.shape[0])]
        y = torch.stack(ys, dim=0)
        if self.bias is not None:
            y = y + self.bias
        return y


# Replace fc1 and f2 forward with custom forward using triton_mvm
def replace_linear_modules(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            new = CustomLinear(child.in_features, child.out_features, bias=(child.bias is not None))
            new.load_state_dict(child.state_dict())
            new.to(device=child.weight.device, dtype=child.weight.dtype)
            new.train(child.training)
            setattr(module, name, new)
        else:
            replace_linear_modules(child)


replace_linear_modules(model)
model = model.cuda()
x = x.cuda()
opt_model = torch.compile(model, backend="inductor")
y = opt_model(x)

# Compare outputs
y = y.cpu()
max_err = (y_ref_model - y).abs().max().item()
print("max abs error TinyMLP with custom mvm:", max_err)
