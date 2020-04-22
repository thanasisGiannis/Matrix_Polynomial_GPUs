#ifndef MATRIXFUNCTIONS_H
#define MATRIXFUNCTIONS_H
#include "cublasparallel.h"
#include <cublas_v2.h>
#include <cuda.h>
#include "strassen.h"

#define ERROR 1
#define OK 0

/* CPU functions */
extern int updateDiag(double *,int,double *,int,double,int,int); /* B = a*I + A */ 
extern int updateMatrix(double *,int,double* ,int,double,int ,int); /* B = a*A */
extern int divisor(int ); /* given an int, returns a primitive int */
extern void matrixMul(double* ,int ,double* ,int ,double* ,int,int,int ,int); /* C = A*B */
extern void initializeZero(double*,int,int,int); /* A = 0*A */
extern void matrixPol(double*,int,double*,int,int,int,double*,int); /* B = f(A,coef) */
extern void matrixAdd(double*,int,double*,int,double*,int,int,int); /* C = A+B */
/* GPU functions */
extern __global__ void setValue(double *,int); /* v[i] = a , for all i */


#endif
