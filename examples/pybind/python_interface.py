import torch
import brevitas.nn as qnn
import mymatmul

matmul_layer = qnn.QuantLinear(in_features=1024, out_features=1024, bias=False)

# Execute matmul in Python
x = torch.randn(1, 1024)
y = matmul_layer(x)

# Execute matmul in C++ using ATen lib
y_cpp = mymatmul.matmul_linear(x, matmul_layer.weight)

print("Output:", y)
print("Output (C++):", y_cpp)
print("y == y_cpp:", torch.allclose(y, y_cpp, atol=1e-2))

# Execute matmul using own cuda kernel
device = torch.device("cuda")
x_cuda = x.to(device, non_blocking=True).contiguous()
W_cuda = matmul_layer.weight.to(device, non_blocking=True).contiguous()
y_cuda = mymatmul.matmul_linear_cuda(x_cuda, W_cuda)

print("Output (cuda):", y_cuda)
print("y == y_cuda:", torch.allclose(y, y_cuda.cpu(), atol=1e-2))
