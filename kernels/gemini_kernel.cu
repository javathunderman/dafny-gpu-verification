#include <iostream>
#include <iomanip>
#include <cmath>

// Define the size of the matrices (N x N)
#define N 1024
// Define the size of the thread block (e.g., 32x32)
#define TILE_WIDTH 32

/**
 * Naive Matrix Multiplication Kernel (C = A * B)
 * Each thread calculates one element of the resulting matrix C.
 *
 * @param A Pointer to the input matrix A (MxK)
 * @param B Pointer to the input matrix B (KxN)
 * @param C Pointer to the output matrix C (MxN)
 * @param M Row dimension of A and C
 * @param K Column dimension of A, and row dimension of B
 * @param N Column dimension of B and C
 */
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int M, int K) {
    // Calculate the row and column index for the current element in C
    // The grid is 2D: gridDim.x * gridDim.y blocks
    // blockIdx.x gives the block column index
    // threadIdx.x gives the thread column index within the block

    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    // Check bounds: only compute if the thread index is within the matrix dimensions
    if (row < M && col < N) {
        /** row = M */

        /** col = N */
        float sum = 0.0f;
        // Perform the dot product: C[row][col] = sum(A[row][k] * B[k][col])
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Host function to execute the matrix multiplication on the GPU.
 */
void launch_matrix_multiply(const float* h_A, const float* h_B, float* h_C, int M, int K) {
    // 1. Calculate size and required memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Pointers for device (GPU) memory
    float *d_A, *d_B, *d_C;

    // 2. Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // 3. Copy host data (input matrices A and B) to device memory
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 4. Define grid and block dimensions
    // Block dimensions: TILE_WIDTH x TILE_WIDTH (e.g., 32x32)
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    // Grid dimensions: Calculate the number of blocks needed to cover the matrix
    // (M + block.y - 1) / block.y ensures ceiling division for grid.y
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    std::cout << "Launching kernel with Grid: (" << grid.x << ", " << grid.y << ") blocks, "
              << "Block: (" << block.x << ", " << block.y << ") threads." << std::endl;

    // 5. Launch the kernel
    // The <<<grid, block>>> syntax specifies the execution configuration
    matrix_multiply_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        // Proceed to cleanup, but the results will be garbage
    }

    // 6. Copy the result matrix C back from device memory to host memory
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 7. Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for post-copy errors (e.g., out of memory errors during the run)
    err = cudaDeviceSynchronize(); // Synchronize to ensure all work is done and errors are caught
    if (err != cudaSuccess) {
        std::cerr << "CUDA execution failed: " << cudaGetErrorString(err) << std::endl;
    }
}

/**
 * Simple host-side sequential CPU implementation for verification.
 */
void cpu_matrix_multiply(const float* A, const float* B, float* C, int M, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Main function to initialize data, run GPU, verify, and print results.
 */
int main() {
    // Set up problem for M x K * K x N multiplication, where M=K=N
    const int M = N, K = N;
    /** requires M == N */
    /** requires K == N */
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C_gpu = M * N * sizeof(float);
    size_t size_C_cpu = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_gpu = (float*)malloc(size_C_gpu);
    float* h_C_cpu = (float*)malloc(size_C_cpu); // For CPU verification

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return 1;
    }

    // Initialize matrices A and B with some data
    std::cout << "Initializing matrices " << M << "x" << K << " and " << K << "x" << N << "..." << std::endl;
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand() % 10); // Random values [0, 9]
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand() % 10);
    }

    // --- 1. Run on GPU ---
    launch_matrix_multiply(h_A, h_B, h_C_gpu, M, K);

    // --- 2. Run on CPU for verification ---
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, M, K);

    // --- 3. Verification ---
    int error_count = 0;
    float max_error = 0.0f;
    const float tolerance = 1e-4f;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > tolerance) {
            error_count++;
        }
        if (diff > max_error) {
            max_error = diff;
        }
    }

    if (error_count > 0) {
        std::cout << "\nVerification FAILED! Total errors: " << error_count << " (Max difference: " << max_error << ")" << std::endl;
    } else {
        std::cout << "\nVerification PASSED! The GPU result matches the CPU result." << std::endl;
    }

    // Optional: Print small matrices for visual check
    if (N <= 10) {
        std::cout << "\n--- Result Matrix C (GPU) ---\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << std::setw(6) << h_C_gpu[i * N + j] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "\nMatrix size is too large (" << N << "x" << N << ") to print result." << std::endl;
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return 0;
}