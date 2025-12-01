// CUDA:
// __global__
// void prefix_sum(const int* in, int* out, int n)
// {
method prefix_sum_thread(
    inArr: array<int>, outArr: array<int>, temp: array<int>,
    n: nat,
    blockDim_x: nat, threadIdx_x: nat
)
  // Arrays must exist
  requires inArr != null && outArr != null && temp != null

  // n is the number of valid elements
  requires n > 0
  requires inArr.Length >= n
  requires outArr.Length >= n

  // Bounds on CUDA thread/block indices
  requires blockDim_x > 0
  requires threadIdx_x < blockDim_x

  // Match the simple kernel assumption: one block, blockDim.x >= n
  requires n <= blockDim_x

  // Shared memory big enough for all threads in the block
  requires temp.Length >= blockDim_x

  modifies outArr, temp
{
  // CUDA: int tid = threadIdx.x;
  var tid: nat := threadIdx_x;

  // Single assert that, with the requires above, guarantees:
  //  - temp[tid] is safe (0 <= tid < blockDim_x <= temp.Length)
  //  - inArr[tid] is safe whenever tid < n (n <= inArr.Length)
  //  - outArr[tid] is safe whenever tid < n (n <= outArr.Length)
  //  - temp[tid - offset] is safe when tid >= offset:
  //       tid < blockDim_x and temp.Length >= blockDim_x
  //       => tid - offset <= blockDim_x - 1 < temp.Length
  assert 0 <= tid < blockDim_x
      && blockDim_x <= temp.Length
      && n <= inArr.Length
      && n <= outArr.Length;

  // CUDA: // Load input into shared memory
  // CUDA: if (tid < n) {
  // CUDA:     temp[tid] = in[tid];
  // CUDA: } else {
  // CUDA:     temp[tid] = 0;
  // CUDA: }
  if tid < n {
    temp[tid] := inArr[tid];
  } else {
    temp[tid] := 0;
  }

  // CUDA: __syncthreads();
  // (no-op in single-thread logical model)

  // CUDA: // Hillis–Steele inclusive scan
  // CUDA: for (int offset = 1; offset < n; offset <<= 1) {
  var offset: nat := 1;
  while offset < n
    invariant 1 <= offset <= 2*n
    // (we don't try to prove algorithmic correctness here,
    // just keep offset in a sane range)
  {
    // CUDA:     int val = 0;
    var val: int := 0;

    // CUDA:     if (tid >= offset && tid < n) {
    // CUDA:         val = temp[tid - offset];
    // CUDA:     }
    if tid >= offset && tid < n {
        assert(tid-offset<temp.Length);
      val := temp[tid - offset];
    }

    // CUDA:     __syncthreads();
    // (no-op)

    // CUDA:     if (tid < n) {
    // CUDA:         temp[tid] += val;
    // CUDA:     }
    if tid < n {
        assert(tid<temp.Length);
      temp[tid] := temp[tid] + val;
    }

    // CUDA:     __syncthreads();
    // (no-op)

    // CUDA:     offset <<= 1;
    offset := offset * 2;
  }

  // CUDA: // Write result
  // CUDA: if (tid < n) {
  // CUDA:     out[tid] = temp[tid];
  // CUDA: }
  if tid < n {
    assert(tid<outArr.Length);
    outArr[tid] := temp[tid];
  }
  // CUDA: }
}
