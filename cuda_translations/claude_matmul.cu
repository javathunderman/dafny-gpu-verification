#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
// C = A * B
__global__ void matmul_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024;  // Matrix size (N x N)
    size_t bytes = N * N * sizeof(float);
    /** requires M == N */
    /** requires K == N */
    /** requires N == 1024 */
    /** requires blockDim_x == 16 */
    /** requires gridDim_x == (N + blockDim_x - 1) */
    /** requires blockDim_y == 16 */
    /** requires gridDim_y == (N + blockDim_y - 1) */
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices with some values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 threads(16, 16);  // 16x16 thread block
    dim3 blocks((N + threads.x - 1) / threads.x, 
                (N + threads.y - 1) / threads.y);
    
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result (optional)
    printf("C[0] = %f (expected %f)\n", h_C[0], (float)N * 2.0f);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}