#include "vision.h"
#include <stdio.h>

void sgbm(float * x, int i, int j)
{
   int n = 2;
   int m = 3;
   int N = 60;

   for(int k = 0; k < 60; k++) {
      printf("X[%d][%d][%d] = %f\n", i, j, k, x[i*m*N + j*N + k]);
   }
}
