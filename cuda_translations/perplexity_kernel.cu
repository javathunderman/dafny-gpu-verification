#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// C = A (MxK) * B (KxN), producing C (MxN)
__global__ void matmul(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* C,
                       int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // col in C

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Matrix sizes
    /** requires M == 4 */
    /** requires K == 3 */
    /** requires N == 5 */
    /** requires blockDim_x == 16 */
    /** requires gridDim_x == 4 */
    /** requires blockDim_y == 16 */
    /** requires gridDim_y == 4 */
    int M = 4;  // rows of A and C
    int K = 3;  // cols of A, rows of B
    int N = 5;  // cols of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize A and B with some values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i + 1);  // 1, 2, 3, ...
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>((i % 7) + 1);  // some pattern
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, sizeA);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_B, sizeB);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_C, sizeC);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy host data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Launch kernel
    matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for device to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print result matrix (small example)
    printf("Result matrix C (%d x %d):\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%6.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
