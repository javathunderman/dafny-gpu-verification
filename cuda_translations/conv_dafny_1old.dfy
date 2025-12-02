// // CUDA:
// // __global__
// // void conv1d_valid(const float* x, const float* h,
// //                   float* y, int N, int K)
// // {
// method conv1d_valid_thread(
//     x: array<real>, h: array<real>, y: array<real>,
//     N: nat, K: nat,
//     blockIdx_x: nat, blockDim_x: nat, threadIdx_x: nat
// )
//   // Arrays allocated in host code
//   requires x != null && h != null && y != null
//   requires x.Length == N
//   requires h.Length >= K          // at least K elements (filter)
//   requires N >= K                 // so N - K + 1 is non-negative
//   requires y.Length >= N - K + 1  // at least outSize elements

//   // Bounds on CUDA thread/block indices
//   requires blockDim_x > 0
//   requires threadIdx_x < blockDim_x
//   // You can optionally bound blockIdx_x too, but it's not needed
//   // for memory safety as long as we don't *assume* anything about i.
//   modifies y
// {
//   // CUDA: int i = blockIdx.x * blockDim.x + threadIdx.x;
//   var i: nat := blockIdx_x * blockDim_x + threadIdx_x;

//   // CUDA: int outSize = N - K + 1;
//   var outSize: nat := N - K + 1;

//   // CUDA:
//   // if (i < outSize) {
//   if i < outSize {
//     // Single assert that, together with the preconditions and guard,
//     // guarantees all array accesses are within bounds:
//     //
//     //  - x[i + j]: j ranges 0 .. K-1, so need i + K - 1 < N == x.Length
//     //  - h[j]: j < K <= h.Length
//     //  - y[i]: i < outSize <= y.Length
//     // assert
//     //   0 <= i
//     //   && i + K - 1 < N
//     //   && K <= h.Length
//     //   && outSize <= y.Length;

//     // CUDA: float acc = 0.0f;
//     var acc: real := 0.0;

//     // CUDA: for (int j = 0; j < K; ++j) {
//     var j: nat := 0;
//     while j < K
//       invariant 0 <= j <= K
//     //   invariant i < outSize ==> 0 <= i //<- can't use bc diff to extract
//       // The assert above, plus these invariants, ensures:
//       //  - 0 <= i + j <= i + K - 1 < N  (safe for x[i + j])
//       //  - 0 <= j < K <= h.Length       (safe for h[j])
//     {
//       // CUDA:     acc += x[i + j] * h[j];
//       assert(i+j<x.Length);
//       assert(j<h.Length);
//       acc := acc + x[i + j] * h[j];

//       // CUDA: }
//       j := j + 1;
//     }

//     // CUDA:     y[i] = acc;
//     assert(i<y.Length);
//     y[i] := acc;
//   }
//   // CUDA: }
// }
