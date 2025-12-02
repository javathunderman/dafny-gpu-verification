// just_lin_expr.c
//  #include <assert.h>

int nondet_int(); // ESBMC's nondeterministic integer

void main() {
  int M   = nondet_int();
  int K   = nondet_int();
  int N   = nondet_int();
  int in1 = nondet_int();

  __ESBMC_assume(in1 > 0);
  __ESBMC_assume(M > 1);
  __ESBMC_assume(K > 0);
  __ESBMC_assume(N > 0);
  __ESBMC_assume(in1 < M);

  __ESBMC_assume(in1 < 1024);
  __ESBMC_assume(M < 1024);
  __ESBMC_assume(K < 1024);
  __ESBMC_assume(N < 1024);

  int row = in1;

  //option 1
  int idx = (row) * K + (K - 1);

  //option 2
  // int idx = (M-1) * K + (K - 1);

  // equivalent Dafny assert:
  assert(idx < M * K);

  //as a sanity check; verify that row is in fact at most M-1
  // assert(row<=M-1);
}
