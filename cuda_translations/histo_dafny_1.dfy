// CUDA:
// __global__
// void histogram(const int* data, int* hist,
//                int N, int numBins)
method histogram_thread(
    data: array<int>, hist: array<int>,
    N: nat, numBins: nat,
    blockIdx_x: nat, blockDim_x: nat, threadIdx_x: nat
)
  // Arrays allocated in host code
  requires data != null && hist != null
  requires data.Length == N
  requires hist.Length == numBins

  // Bounds on CUDA thread/block indices
  requires blockDim_x > 0
  requires threadIdx_x < blockDim_x

  modifies hist
{
  // CUDA: int i = blockIdx.x * blockDim.x + threadIdx.x;
  var i: nat := blockIdx_x * blockDim_x + threadIdx_x;

  // CUDA: if (i < N) {
  if i < N {
    // CUDA:     int bin = data[i];
    var bin: int := data[i];

    // Single assert: with the preconditions and guards, this
    // ensures all array accesses are in-bounds:
    //
    //  - data[i]: we have i < N == data.Length
    //  - hist[bin]: we only access hist[bin] when
    //       0 <= bin && bin < numBins == hist.Length
    assert
      (i < N ==> i < data.Length) &&
      ((0 <= bin && bin < numBins) ==> (0 <= bin && bin < hist.Length));

    // CUDA:     if (0 <= bin && bin < numBins) {
    // CUDA:         atomicAdd(&hist[bin], 1);
    // CUDA:     }
    if 0 <= bin && bin < numBins {
      // Model atomicAdd as a simple increment in the single-thread view
      hist[bin] := hist[bin] + 1;
    }
  }
  // CUDA: }
}
