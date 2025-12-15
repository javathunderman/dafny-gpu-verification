#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Dimensions for the matrices A(MxK), B(KxN), C(MxN)
const int M = 64;
const int N = 64;
const int K = 64;

// The kernel definition from above would go here or in a separate header.
// ... (The matrixMulKernel from section 1)

// Helper function to check for CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
// Kernel for matrix multiplication C = A * B
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate the row and column of the result matrix C element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure we don't access memory outside the matrix dimensions
    if (row < M && col < N) {
        float sum = 0.0f;
        // The core matrix multiplication logic: C[row][col] = sum(A[row][k] * B[k][col])
        for (int k = 0; k < K; ++k) {
            // A is row-major: A[row * K + k]
            // B is row-major: B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }
        // Store the result in C
        C[row * N + col] = sum;
    }
}
int main() {
    // --- Host Memory Allocation and Initialization ---
    /** requires M == N */
    /** requires K == N */
    /** requires N == 64 */
    /** requires blockDim_x == 16 */
    /** requires gridDim_x == (N + blockDim_x - 1) */
    /** requires blockDim_y == 16 */
    /** requires gridDim_y == (N + blockDim_y - 1) */
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA); // Host memory for A
    float *h_B = (float*)malloc(sizeB); // Host memory for B
    float *h_C = (float*)malloc(sizeC); // Host memory for result C

    // Initialize A and B matrices (simplified, for demonstration)
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;
    // (If A=64x64, B=64x64 and all elements are 1 and 2, then C elements will be 64*1*2 = 128)

    // --- Device Memory Allocation ---
    float *d_A, *d_B, *d_C; // Device pointers
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    // --- Host to Device (H2D) Data Transfer ---
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // --- Kernel Configuration and Launch ---
    int TILE_SIZE = 16;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE); // 16x16 threads per block
    
    // Calculate required grid dimensions. 
    // Uses a ceil division formula: (dimension + TILE_SIZE - 1) / TILE_SIZE
    dim3 dimGrid(
        (N + dimBlock.x - 1) / dimBlock.x, // Grid size in X (columns of C)
        (M + dimBlock.y - 1) / dimBlock.y  // Grid size in Y (rows of C)
    );

    printf("Launching kernel with Grid Size: (%d, %d) and Block Size: (%d, %d)\n",
           dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    // Launch the kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Synchronize to wait for the kernel to complete
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Check for errors that might have occurred during kernel execution
    CHECK_CUDA(cudaGetLastError());

    // --- Device to Host (D2H) Data Transfer ---
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // --- Verification and Cleanup ---
    
    // Minimal verification (Check the first element)
    float expected_C00 = (float)K * 1.0f * 2.0f; // K * (A element) * (B element)
    if (h_C[0] == expected_C00) {
        printf("Success! C[0][0] = %.0f (Expected: %.0f)\n", h_C[0], expected_C00);
    } else {
        printf("Error! C[0][0] = %.0f (Expected: %.0f)\n", h_C[0], expected_C00);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

 
   return 0;
}
