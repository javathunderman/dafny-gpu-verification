// Axiomatic lemma for 2D-to-1D linear indexing (like a row-major layout).
lemma {:axiom} LinearIndex2DBound(row: nat, col: nat, H: nat, W: nat)
  requires row < H
  requires col < W
  ensures row * W + col < H * W

// CUDA:
// __global__
// void conv2d_valid(const float* img,
//                   const float* filt,
//                   float* out,
//                   int H, int W,
//                   int KH, int KW)
method conv2d_valid_thread(
    img: array<int>,
    filt: array<int>,
    out: array<int>,
    H: nat, W: nat,     // image height, width
    KH: nat, KW: nat,   // kernel height, width
    blockIdx_x: nat, blockIdx_y: nat,
    blockDim_x: nat, blockDim_y: nat,
    threadIdx_x: nat, threadIdx_y: nat
)
  // Arrays must exist
  requires img != null && filt != null && out != null

  // Image and filter sizes
  requires H > 0 && W > 0
  requires KH > 0 && KW > 0
  requires H >= KH && W >= KW

  // Flattened sizes
  requires img.Length == H * W
  requires filt.Length == KH * KW

  // Output size (valid conv)
  // out has size (H-KH+1) * (W-KW+1)
  requires out.Length == (H - KH + 1) * (W - KW + 1)

  // Bounds on CUDA block and thread indices
  requires blockDim_x > 0 && blockDim_y > 0
  requires threadIdx_x < blockDim_x
  requires threadIdx_y < blockDim_y

  modifies out
{
  // CUDA:     int outH = H - KH + 1;
  // CUDA:     int outW = W - KW + 1;
  var outH: nat := H - KH + 1;
  var outW: nat := W - KW + 1;

  // CUDA:     int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  // CUDA:     int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  var outRow: nat := blockIdx_y * blockDim_y + threadIdx_y;
  var outCol: nat := blockIdx_x * blockDim_x + threadIdx_x;

  // CUDA:     if (outRow < outH && outCol < outW) {
  if outRow < outH && outCol < outW {

    var acc: int := 0;


    var kr: nat := 0;
    while kr < KH
      invariant 0 <= kr <= KH
    {
      var kc: nat := 0;
      while kc < KW
        invariant 0 <= kc <= KW
      {

        var inRow: nat := outRow + kr;
        var inCol: nat := outCol + kc;


        // --- apply trick    ---
        //assert(inRow * W + inCol < H*W);
        //doesn't assert? replace vars (non pure inputs)

        // assert((outRow + kr) * W + inCol < H*W);
        //doesn't assert? replace vars (non pure inputs)
        
        // assert((outRow + kr) * W + (outCol + kc) < H*W);
        //doesn't assert? replace vars (non pure inputs)
        
        // assert(((outH-1) + kr) * W + ((outW-1) + kc) < H*W);
        //doesn't assert? reaplace using control fl heuristic (if outRow < outH && outCol < outW {...)
        
        // assert((((H - KH + 1)-1) + kr) * W + (((W - KW + 1)-1) + kc) < H*W);
        //doesn't assert? replace using control flow heuristic
          //while kr < KH...
          //while kc < KW
        
        assert((((H - KH + 1)-1) + (KH-1)) * W + (((W - KW + 1)-1) + (KW-1)) < H*W);
        //var imgVal: int := img[inRow * W + inCol];

        kc := kc + 1;
      }

      kr := kr + 1;
    }

  }

}
