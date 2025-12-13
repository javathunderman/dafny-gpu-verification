// tri_packed_read.cu
// Minimal CUDA program: packed upper-triangular (including diagonal) stored in 1D.
// Each thread reads one (row,col) pair, computes idx = row*(row+1)/2 + col, and loads packed[idx].

#include <cstdio>
#include <cuda_runtime.h>

__global__ void TriPackedReadKernel(const int* packed, int packedLen,
                                   const int* rows, const int* cols,
                                   int n, int* out, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        int row = rows[t];
        int col = cols[t];

        // In a real kernel you might omit these checks for performance and rely on guarantees.
        // Keeping them here makes the example robust.
        if (0 <= col && col <= row && row < n) {
            int base = (row * (row + 1)) / 2;   // triangular number
            int idx  = base + col;

            if (0 <= idx && idx < packedLen) {
                out[t] = packed[idx];
            } else {
                out[t] = -777777; // indicates out-of-bounds (shouldn't happen if inputs are valid)
            }
        } else {
            out[t] = -999999; // indicates invalid (row,col)
        }
    }
}

static void checkCuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main()
{
    // Choose a small n so it’s easy to inspect.
    const int n = 6;
    const int packedLen = n * (n + 1) / 2;

    // Host-side packed upper-triangular values: packed[idx] = idx (easy to validate)
    int h_packed[packedLen];
    for (int i = 0; i < packedLen; ++i) h_packed[i] = i;

    // We'll launch threads, each reading some (row,col).
    // Make sure 0 <= col <= row < n.
    const int T = 12;
    int h_rows[T] = {0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5};
    int h_cols[T] = {0, 0, 1, 0, 1, 2, 0, 3, 0, 4, 0, 5};

    int h_out[T];
    for (int i = 0; i < T; ++i) h_out[i] = -1;

    // Device allocations
    int *d_packed = nullptr, *d_rows = nullptr, *d_cols = nullptr, *d_out = nullptr;
    checkCuda(cudaMalloc(&d_packed, packedLen * sizeof(int)), "cudaMalloc d_packed");
    checkCuda(cudaMalloc(&d_rows,   T * sizeof(int)),        "cudaMalloc d_rows");
    checkCuda(cudaMalloc(&d_cols,   T * sizeof(int)),        "cudaMalloc d_cols");
    checkCuda(cudaMalloc(&d_out,    T * sizeof(int)),        "cudaMalloc d_out");

    // Copy inputs
    checkCuda(cudaMemcpy(d_packed, h_packed, packedLen * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy packed H2D");
    checkCuda(cudaMemcpy(d_rows, h_rows, T * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy rows H2D");
    checkCuda(cudaMemcpy(d_cols, h_cols, T * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy cols H2D");

    // Launch
    dim3 block(128);
    dim3 grid((T + block.x - 1) / block.x);
    TriPackedReadKernel<<<grid, block>>>(d_packed, packedLen, d_rows, d_cols, n, d_out, T);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy output back
    checkCuda(cudaMemcpy(h_out, d_out, T * sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy out D2H");

    // Print results
    std::printf("n=%d packedLen=%d\n", n, packedLen);
    for (int t = 0; t < T; ++t) {
        int row = h_rows[t], col = h_cols[t];
        int idx = (row * (row + 1)) / 2 + col;
        std::printf("t=%2d (row=%d,col=%d) idx=%2d  packed[idx]=%d  out=%d\n",
                    t, row, col, idx, h_packed[idx], h_out[t]);
    }

    cudaFree(d_packed);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_out);
    return 0;
}
