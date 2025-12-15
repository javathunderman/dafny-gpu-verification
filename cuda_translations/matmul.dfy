lemma {:axiom} LinearIndexBound(row: nat, col: nat, M: nat, N: nat)
  requires row < M
  requires col < N
  ensures row * N + col < M * N

method matMulKernelThread(
    A: array<int>,
    B: array<int>,
    C: array<int>,
    M: nat, K: nat, N: nat,
    blockIdx_x: nat, blockIdx_y: nat,
    blockDim_x: nat, blockDim_y: nat,
    gridDim_x: nat, gridDim_y: nat,
    threadIdx_x: nat, threadIdx_y: nat
)
  // Arrays allocated in host code
  requires A != null && B != null && C != null
  requires A.Length == M * K
  requires B.Length == K * N
  requires C.Length == M * N

  // threadIdx bounded by threadcount
  requires 0 <= threadIdx_x && threadIdx_x < blockDim_x
  requires 0 <= threadIdx_y && threadIdx_y < blockDim_y

  // blockIdx bounded by blockcount
  requires 0 <= blockIdx_x && blockIdx_x < gridDim_x
  requires 0 <= blockIdx_y && blockIdx_y < gridDim_y

  // // This method is allowed to modify array C
  modifies C
{
  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  var row := blockIdx_y * blockDim_y + threadIdx_y;

  // int col = blockIdx.x * blockDim.x + threadIdx.x;
  var col := blockIdx_x * blockDim_x + threadIdx_x;

  // if (row >= M || col >= N) return;
  if row >= M || col >= N {
    return;
  }

  // float sum = 0.0f;
  var sum: int := 0;

  // for (int k = 0; k < K; ++k) {
  var k := 0;
  while k < K
    invariant 0 <= k <= K
  {
    // index calculations
    var idxA := (M-1) * K + k;
    
    //var idxB := k * N + (N-1);
    var idxB := (K-1)*N + (N-1);

    // INLINE ASSERTIONS (exactly next to each access)

    assert 0 <= idxA < A.Length;          // A[idxA]
    assert 0 <= idxB < B.Length;          // B[idxB]

    //   sum += A[row * K + k] * B[k * N + col];
    sum := sum + A[idxA] * B[idxB];

    k := k + 1;
  }

  var idxC := row * N + col;


  LinearIndexBound(row, col, M, N);

  //INLINE ASSERTION next to C write
  assert 0 <= idxC < C.Length;

  //C[row * N + col] = sum;
  C[idxC] := sum;
}