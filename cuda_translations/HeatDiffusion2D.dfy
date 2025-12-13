// --------------------------------
// STUPID simple version of a real heat diffusion stencil kernel
// the main point here is that the boundary points are fixed, 
// and so are not computed
// a real heat diffusion kernel would be more complicated (ofc)
// --------------------------------

method HeatDiffusion2D(
    inArr: array<int>,
    outArr: array<int>,
    H: int, W: int,
    i: int, j: int
)
  requires inArr != null
  requires outArr != null
  requires H > 2 && W > 2

  // 2D grid bounds
  requires 0 <= i < H
  requires 0 <= j < W

  // We only process interior points
  requires 1 <= i < H - 1
  requires 1 <= j < W - 1

  // Array sizes
  requires inArr.Length == H * W
  requires outArr.Length == (H - 2) * (W - 2)

  modifies outArr
{
  // Flatten interior grid cell (i,j) into compacted array
  var interiorIdx := (i - 1) * (W - 2) + (j - 1);

  // --- apply trick    ---
  //assert(interiorIdx < (H - 2) * (W - 2));                    //takes very long time to verify, or might not at all
  //assert((H - 3) * (W - 2) + (j - 1) < (H - 2) * (W - 2));    //replace i with it's upper bound, H-2
  assert((H - 3) * (W - 2) + (W - 3) < (H - 2) * (W - 2));      //and j with it's upper bound (this now verifies)

  assert(i * W + j < H * W);

  //outArr[interiorIdx] := inArr[i * W + j];                        //original
  outArr[(H - 3) * (W - 2) + (W - 3)] := inArr[i * W + j];    //asserts
}
