import torch
import triton
from src.gemm_kernel import check_and_launch_matmul, matmul_kernel
from src.print_stats import print_tuning_stats

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Test
torch.manual_seed(0)
a = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
triton_output = check_and_launch_matmul(a, b)
torch_output = torch.matmul(a, b)

# Compare outputs
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
    print("Test passed! ✅")
    print_tuning_stats(matmul_kernel)
else:
    print("Test failed! ❌")
