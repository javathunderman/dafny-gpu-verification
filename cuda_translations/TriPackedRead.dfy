// -----------------------------------
// kind of a contrived kernel, but this takes an upper left triangle matrix and packs it into an array
// this minimal function just gets the original row / col out of the 1D array
// jsut complex enough to apply the variable trick
// -----------------------------------

method TriPackedRead(packed: array<int>, outArr: array<int>, n: int, row: int, col: int)
  requires packed != null && outArr != null
  requires n > 0
  requires 0 <= col <= row < n
  // packed stores the upper triangle (including diagonal), length = n*(n+1)/2
  requires packed.Length == n * (n + 1) / 2
  requires outArr.Length >= 1
  modifies outArr
{
  // base index for row in packed triangular layout
  var base := row * (row + 1) / 2;
  var idx  := base + col;


  // --- apply variable trick    ---
  // again this wont verify 
  outArr[0] := packed[idx];                             //<- doesn't verify
  //outArr[0] := packed[base + col];                    //<- replace idx with it's definition
  //outArr[0] := packed[row * (row + 1) / 2 + col];     //<- replace base with it's definition
  //outArr[0] := packed[(n-1) * ((n-1) + 1) / 2 + col]; //<- this does verify!
}

