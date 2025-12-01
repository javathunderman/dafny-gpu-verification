#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------- CUDA KERNEL ----------------
// Inclusive prefix sum:
// out[i] = in[0] + in[1] + ... + in[i]
//
// Assumptions to keep it simple:
//  - single block (gridDim.x == 1)
//  - blockDim.x >= n
__global__
void prefix_sum(const int* in, int* out, int n)
{
    extern __shared__ int temp[]; // shared memory: size >= n

    int tid = threadIdx.x;

    // Load input into shared memory
    if (tid < n) {
        temp[tid] = in[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Hillis–Steele inclusive scan
    for (int offset = 1; offset < n; offset <<= 1) {
        int val = 0;
        if (tid >= offset && tid < n) {
            val = temp[tid - offset];
        }
        __syncthreads();          // make sure all threads see old temp[]
        if (tid < n) {
            temp[tid] += val;
        }
        __syncthreads();          // make sure all threads see updated temp[]
    }

    // Write result
    if (tid < n) {
        out[tid] = temp[tid];
    }
}

// ---------------- CPU REFERENCE ----------------
void prefix_sum_cpu(const int* in, int* out, int n)
{
    int running = 0;
    for (int i = 0; i < n; ++i) {
        running += in[i];
        out[i] = running;
    }
}

// ---------------- MAIN / HOST CODE ----------------
int main()
{
    const int N = 16; // length of the array

    // Host allocations
    int* h_in  = (int*)malloc(N * sizeof(int));
    int* h_out = (int*)malloc(N * sizeof(int));
    int* h_ref = (int*)malloc(N * sizeof(int));

    // Initialize input with simple values
    for (int i = 0; i < N; ++i) {
        h_in[i] = i + 1;  // 1, 2, 3, ...
    }

    // Device allocations
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: single block, N threads
    int threadsPerBlock = N;       // simple: one thread per element
    int sharedMemBytes  = N * sizeof(int);
    prefix_sum<<<1, threadsPerBlock, sharedMemBytes>>>(d_in, d_out, N);

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute reference on CPU
    prefix_sum_cpu(h_in, h_ref, N);

    // Print and compare
    printf("Input:      ");
    for (int i = 0; i < N; ++i) printf("%3d ", h_in[i]);
    printf("\nGPU prefix: ");
    for (int i = 0; i < N; ++i) printf("%3d ", h_out[i]);
    printf("\nCPU prefix: ");
    for (int i = 0; i < N; ++i) printf("%3d ", h_ref[i]);
    printf("\n");

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_ref[i]) {
            ok = false;
            printf("Mismatch at i=%d: GPU=%d CPU=%d\n",
                   i, h_out[i], h_ref[i]);
        }
    }
    printf("\nResult: %s\n", ok ? "PASS" : "FAIL");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);

    return ok ? 0 : 1;
}
