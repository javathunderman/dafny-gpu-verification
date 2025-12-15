#include <stdio.h>

__global__ void matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    // A is M x N
    // B is N x K
    // C is M x K

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;

        // dot product of row of A and col of B
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }

        C[row * K + col] = sum;
    }
}

int main() {
    int M = 512, N = 512, K = 512;

    // allocate and initialize host memory (example)
    size_t bytesA = M * N * sizeof(float);
    size_t bytesB = N * K * sizeof(float);
    size_t bytesC = M * K * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);

    for (int i = 0; i < M*N; ++i) hA[i] = 1.0f;
    for (int i = 0; i < N*K; ++i) hB[i] = 1.0f;

    // device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    matmul<<<grid, block>>>(dA, dB, dC, M, N, K);

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;
}
