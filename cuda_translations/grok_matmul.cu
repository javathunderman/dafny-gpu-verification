#include <stdio.h>
#include <cuda_runtime.h>

// Device kernel: naive matrix multiplication (one thread per C element)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;  // Matrix size (N x N); change as needed
    const int bytes = N * N * sizeof(float);
    /** requires M == N */
    /** requires K == N */
    /** requires N == 1024 */
    /** requires blockDim_x == 16 */
    /** requires gridDim_x == (N + blockDim_x - 1) */
    /** requires blockDim_y == 16 */
    /** requires gridDim_y == (N + blockDim_y - 1) */
    // Host matrices
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host matrices (example: A = identity, B = ones for simple testing)
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (i / N == i % N) ? 1.0f : 0.0f;  // Identity matrix
        h_B[i] = 1.0f;                           // All ones
    }

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch configuration (16x16 threads per block is common)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Simple verification (for identity * ones = rows of ones)
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        float expected = (float)N;
        if (fabs(h_C[i] - expected) > 1e-3f) {
            correct = false;
            break;
        }
    }
    printf("Result %s\n", correct ? "correct" : "incorrect");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}