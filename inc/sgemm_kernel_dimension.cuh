#ifndef SGEMM_KERNEL_DIMENSION_CUH
#define SGEMM_KERNEL_DIMENSION_CUH

// Kernel 00
constexpr int BLOCKSIZE_00 = 32;

// Kernel 01
constexpr int BLOCKSIZE_01 = 32;

// Kernel 02
constexpr int BLOCKSIZE_02 = 32;

// Kernel 03
constexpr int BM_03 = 64;
constexpr int BN_03 = 32;
constexpr int TM_03 = 4;
constexpr int TN_03 = 2;
constexpr int BK_03 = 16;

// Kernel 04
constexpr int BM_04 = 64;
constexpr int BN_04 = 32;
constexpr int TM_04 = 4;
constexpr int TN_04 = 2;
constexpr int BK_04 = 16;
using DTypeVector_04 = float2;

// Kernel 05
constexpr int BM_05 = 64;
constexpr int BN_05 = 32;
constexpr int TM_05 = 4;
constexpr int TN_05 = 2;
constexpr int BK_05 = 16;
using DTypeVector_05 = float2;

// Kernel 06
constexpr int BM_06 = 64;
constexpr int BN_06 = 32;
constexpr int TM_06 = 4;
constexpr int TN_06 = 2;
constexpr int BK_06 = 16;
constexpr int WN_06 = 16;
constexpr int WM_06 = 16;

// Kernel 07
constexpr int BLOCKSIZE_07 = 16;
constexpr int TM_07 = 8;
constexpr int TN_07 = 1;

// Kernel 10
constexpr int BLOCKSIZE_10 = 32;

// Kernel 11
constexpr int BLOCKSIZE_11 = 32;
constexpr int BLOCKSIZE_X_11 = 64;

#endif // SGEMM_KERNEL_DIMENSION_CUH
