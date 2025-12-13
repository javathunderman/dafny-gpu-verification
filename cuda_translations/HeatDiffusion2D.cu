// heat_diffusion_simple.cu
#include <cstdio>
#include <cuda_runtime.h>

// --------------------------------
// STUPID simple version of a real heat diffusion stencil kernel
// Boundary points are fixed and NOT computed.
// Only interior points are gathered into a compact array.
// --------------------------------
__global__ void HeatDiffusion2D(
    const int* inArr,   // size H * W
    int* outArr,        // size (H - 2) * (W - 2)
    int H, int W
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process interior points
    if (i >= 1 && i < H - 1 &&
        j >= 1 && j < W - 1)
    {
        // Flatten interior grid cell (i,j) into compacted array
        int interiorIdx = (i - 1) * (W - 2) + (j - 1);

        // Access original grid (row-major)
        int inIdx = i * W + j;

        // Write interior value
        outArr[interiorIdx] = inArr[inIdx];
    }
}

int main()
{
    // Small, concrete sizes (easy for verifiers)
    const int H = 6;
    const int W = 7;

    const int inSize  = H * W;
    const int outSize = (H - 2) * (W - 2);

    int h_in[inSize];
    int h_out[outSize];

    // Initialize input grid with something trivial
    for (int i = 0; i < inSize; ++i) {
        h_in[i] = i;
    }

    int *d_in = nullptr;
    int *d_out = nullptr;

    cudaMalloc(&d_in,  inSize  * sizeof(int));
    cudaMalloc(&d_out, outSize * sizeof(int));

    cudaMemcpy(d_in, h_in, inSize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 block(8, 8);
    dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y
    );

    HeatDiffusion2D<<<grid, block>>>(d_in, d_out, H, W);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, outSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print interior array (optional)
    printf("Interior values:\n");
    for (int i = 0; i < outSize; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
