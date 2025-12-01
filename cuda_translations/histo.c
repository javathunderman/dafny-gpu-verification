#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------- CUDA KERNEL ----------------
// Each thread processes one element data[i].
// If the value is in [0, numBins), it atomically increments hist[bin].
__global__
void histogram_kernel(const int* data, int* hist,
                      int N, int numBins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int bin = data[i];
        if (0 <= bin && bin < numBins) {
            atomicAdd(&hist[bin], 1);
        }
    }
}

// ---------------- CPU REFERENCE ----------------
void histogram_cpu(const int* data, int* hist,
                   int N, int numBins)
{
    // Initialize histogram to 0
    for (int b = 0; b < numBins; ++b) {
        hist[b] = 0;
    }
    // Count
    for (int i = 0; i < N; ++i) {
        int bin = data[i];
        if (0 <= bin && bin < numBins) {
            hist[bin]++;
        }
    }
}

// ---------------- MAIN / HOST CODE ----------------
int main()
{
    const int N = 1 << 16;  // number of data elements
    const int numBins = 16; // number of histogram bins

    // Host allocations
    int* h_data   = (int*)malloc(N * sizeof(int));
    int* h_hist   = (int*)malloc(numBins * sizeof(int));
    int* h_hist_ref = (int*)malloc(numBins * sizeof(int));

    if (!h_data || !h_hist || !h_hist_ref) {
        printf("Host malloc failed\n");
        return 1;
    }

    // Initialize data with values in [0, numBins)
    for (int i = 0; i < N; ++i) {
        // Simple pattern; you could also use rand() % numBins.
        h_data[i] = i % numBins;
    }

    // Device allocations
    int *d_data = nullptr, *d_hist = nullptr;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_hist, numBins * sizeof(int));

    if (!d_data || !d_hist) {
        printf("Device malloc failed\n");
        return 1;
    }

    // Copy input data to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device histogram to 0
    cudaMemset(d_hist, 0, numBins * sizeof(int));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching histogram kernel with %d blocks, %d threads/block\n",
           blocks, threadsPerBlock);

    histogram_kernel<<<blocks, threadsPerBlock>>>(d_data, d_hist, N, numBins);

    // Wait for GPU and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_hist, d_hist, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute reference on CPU
    histogram_cpu(h_data, h_hist_ref, N, numBins);

    // Print a small summary
    printf("Histogram (GPU vs CPU):\n");
    bool ok = true;
    for (int b = 0; b < numBins; ++b) {
        printf("  bin %2d: GPU=%6d  CPU=%6d\n", b, h_hist[b], h_hist_ref[b]);
        if (h_hist[b] != h_hist_ref[b]) {
            ok = false;
        }
    }

    printf("\nResult: %s\n", ok ? "PASS" : "FAIL");

    // Clean up
    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    free(h_hist);
    free(h_hist_ref);

    return ok ? 0 : 1;
}
