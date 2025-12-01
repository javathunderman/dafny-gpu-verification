#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------- CUDA KERNEL ----------------
// y[i] = sum_{j=0..K-1} x[i + j] * h[j]
// "valid" convolution: i = 0 .. N-K
__global__
void conv1d_valid(const float* x, const float* h,
                  float* y, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int outSize = N - K + 1;

    if (i < outSize) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            acc += x[i + j] * h[j];
        }
        y[i] = acc;
    }
}

// ---------------- CPU REFERENCE ----------------
void conv1d_valid_cpu(const float* x, const float* h,
                      float* y, int N, int K)
{
    int outSize = N - K + 1;
    for (int i = 0; i < outSize; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            acc += x[i + j] * h[j];
        }
        y[i] = acc;
    }
}

// ---------------- MAIN / HOST CODE ----------------
int main()
{
    const int N = 16;   // input length
    const int K = 3;    // filter length
    const int outSize = N - K + 1;

    // Host allocations
    float *h_x  = (float*)malloc(N * sizeof(float));
    float *h_h  = (float*)malloc(K * sizeof(float));
    float *h_y  = (float*)malloc(outSize * sizeof(float));
    float *h_y_ref = (float*)malloc(outSize * sizeof(float));

    // Initialize input and filter with simple values
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i + 1);   // 1,2,3,...
    }
    for (int j = 0; j < K; ++j) {
        h_h[j] = 1.0f;  // simple [1,1,1] box filter
    }

    // Device allocations
    float *d_x, *d_h, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_h, K * sizeof(float));
    cudaMalloc(&d_y, outSize * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 128;
    int blocks = (outSize + threadsPerBlock - 1) / threadsPerBlock;
    conv1d_valid<<<blocks, threadsPerBlock>>>(d_x, d_h, d_y, N, K);

    // Wait for GPU to finish, check for errors (optional but helpful)
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_y, d_y, outSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference on CPU
    conv1d_valid_cpu(h_x, h_h, h_y_ref, N, K);

    // Compare and print
    printf("Input x: ");
    for (int i = 0; i < N; ++i) printf("%4.1f ", h_x[i]);
    printf("\nFilter h: ");
    for (int j = 0; j < K; ++j) printf("%4.1f ", h_h[j]);
    printf("\n\nGPU output y:     ");
    for (int i = 0; i < outSize; ++i) printf("%6.1f ", h_y[i]);
    printf("\nCPU reference y: ");
    for (int i = 0; i < outSize; ++i) printf("%6.1f ", h_y_ref[i]);
    printf("\n");

    // Check correctness
    bool ok = true;
    for (int i = 0; i < outSize; ++i) {
        if (fabs(h_y[i] - h_y_ref[i]) > 1e-4f) {
            ok = false;
            printf("Mismatch at i=%d: GPU=%f CPU=%f\n",
                   i, h_y[i], h_y_ref[i]);
        }
    }
    printf("\nResult: %s\n", ok ? "PASS" : "FAIL");

    // Clean up
    cudaFree(d_x);
    cudaFree(d_h);
    cudaFree(d_y);
    free(h_x);
    free(h_h);
    free(h_y);
    free(h_y_ref);

    return ok ? 0 : 1;
}
