from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant_tensor import QuantTensor
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


class IntMVM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W_f: torch.Tensor, x_f: torch.Tensor, s_w: torch.Tensor, s_x: torch.Tensor,
                bw_w: int, bw_x: int):
        assert W_f.dim() == 2 and x_f.dim() == 1
        M, K = W_f.shape

        # Zero point s_w can be per-row (vector) or scalar
        if s_w.dim() == 0:
            s_w_vec = s_w.view(1).expand(M)
        else:
            s_w_vec = s_w.view(-1)
            assert s_w_vec.numel() == M
        s_x_scalar = s_x.reshape(-1)[0]

        # Float to integer conversion of weights and inputs
        qW_i32 = torch.round(W_f / s_w_vec.view(-1, 1)).to(torch.int32)
        qX_i32 = torch.round(x_f / s_x_scalar).to(torch.int32)

        # Launch MVM kernel with integer inputs and weights
        y_i32 = torch.ops.mylib.mvm(qW_i32, qX_i32)
        # Integer to float conversion of MVM result
        y = y_i32.to(torch.float32) * (s_w_vec * s_x_scalar)

        # Save for backward pass
        ctx.save_for_backward(W_f, x_f, qW_i32.to(torch.float32), qX_i32.to(torch.float32), s_w_vec,
                              s_x_scalar.unsqueeze(0))
        ctx.bw_w = bw_w
        ctx.bw_x = bw_x
        return y

    @staticmethod
    def backward(ctx, grad_y):
        W_f, x_f, qW_f, qX_f, s_w_vec, s_x_vec = ctx.saved_tensors
        bw_x = ctx.bw_x
        bw_w = ctx.bw_w

        # δL(y)/δW = δL(y)/δy * δy(W)/δW = grad_y * x
        dW = grad_y.unsqueeze(1) * x_f.unsqueeze(0)
        # δL(y)/δx = δL(y)/δy * δy(x)/δx = grad_y * W
        dx = W_f.transpose(0, 1).matmul(grad_y)

        # δL(y)/δs_x = grad_y * s_w * [qw * qx - qw * x / s_x]
        # Mask used for clipping function (STE)
        mask_x = (qX_f.abs() < (2**(bw_x - 1))).to(x_f.dtype)
        qw_qx = (qW_f * qX_f.unsqueeze(0)).sum(dim=1)
        qw_x = (W_f * (mask_x * x_f).unsqueeze(0)).sum(dim=1)
        ds_x = (grad_y * (s_w_vec * (qw_qx - qw_x / s_x_vec[0]))).sum().unsqueeze(0)

        # δL(y)/δs_w = grad_y * s_x * [qw * qx - qW * W / s_w]
        mask_w = (qW_f.abs() < (2**(bw_w - 1))).to(W_f.dtype)
        W_qx = (mask_w * W_f * qX_f.unsqueeze(0)).sum(dim=1)
        ds_w_row = grad_y * (s_x_vec[0] * (qw_qx - W_qx / s_w_vec))

        # Return gradients w.r.t. inputs of forward()
        return dW, dx, ds_w_row, ds_x, None, None


class TinyMLP(torch.nn.Module):

    def __init__(self, in_features=128, hidden=64, out_features=10):
        super().__init__()
        self.fc1 = qnn.QuantLinear(in_features,
                                   hidden,
                                   bias=True,
                                   input_quant=Int8ActPerTensorFloat,
                                   return_quant_tensor=True)
        self.act = torch.nn.ReLU()
        self.fc2 = qnn.QuantLinear(hidden,
                                   out_features,
                                   bias=True,
                                   input_quant=Int8ActPerTensorFloat,
                                   return_quant_tensor=True)

    def forward(self, x):
        x = self.fc1(x)
        if isinstance(x, QuantTensor): x = x.tensor
        x = self.act(x)
        x = self.fc2(x)
        if isinstance(x, QuantTensor): x = x.tensor
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
        kwargs.setdefault("return_quant_tensor", True)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        assert x.dim() == 2

        # Quantize inputs and weights
        q_in = self.input_quant(x)
        q_w = self.quant_weight(q_in)
        assert q_in.zero_point == 0
        assert q_w.zero_point == 0

        # Perform MVM for each batch element using Triton kernel with STE backward
        ys = []
        for b in range(q_in.size(0)):
            y_b = IntMVM.apply(q_w.tensor, q_in.tensor[b], q_w.scale, q_in.scale, q_w.bit_width,
                               q_in.bit_width)
            ys.append(y_b)
        y = torch.stack(ys, dim=0)

        # Add bias and quantize output
        if self.bias is not None:
            y = y + self.bias
        q_out = self.output_quant(y)
        return q_out


# Replace fc1 and f2 forward with custom forward using triton_mvm
def replace_brevitas_linear_with_int_triton(module: torch.nn.Module,
                                            return_quanttensor: bool = False):
    for name, child in list(module.named_children()):
        if isinstance(child, qnn.QuantLinear):
            new = QuantLinearIntTriton(in_features=child.in_features,
                                       out_features=child.out_features,
                                       bias=(child.bias is not None),
                                       weight_quant=getattr(child, "weight_quant", None),
                                       bias_quant=getattr(child, "bias_quant", None),
                                       input_quant=getattr(child, "input_quant", None),
                                       output_quant=getattr(child, "output_quant", None),
                                       return_quant_tensor=True)
            new.load_state_dict(child.state_dict(), strict=False)
            new.to(device=next(child.parameters()).device, dtype=next(child.parameters()).dtype)
            new.train(child.training)
            setattr(module, name, new)
        else:
            replace_brevitas_linear_with_int_triton(child, return_quanttensor=return_quanttensor)


import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

batch_size = 128
in_features = 28 * 28
tf = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.view(-1))])
train_set = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=tf)
test_set = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        avg_loss += loss.item() * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
    return avg_loss / total, correct / total


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyMLP(in_features=in_features, hidden=256, out_features=10).to(device)
replace_brevitas_linear_with_int_triton(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 5
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = evaluate(model, test_loader, device)
    print(
        f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
    )

if not os.getenv("TRITON_INTERPRET") == "1":
    model = torch.compile(model, backend="inductor")

int_loss, int_acc = evaluate(model, test_loader, device)
print(f"[INT-EVAL] val_loss={int_loss:.4f}  val_acc={int_acc*100:.2f}%")
