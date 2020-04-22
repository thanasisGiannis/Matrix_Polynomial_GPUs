#ifndef STRASSEN_H
#define STRASSEN_H


#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <cmath>


void GPU_mul(cublasHandle_t handle, double *A, double *B, double *C,
	int lda, int ldb, int ldc,
	int XA, int XB, int XC,
	int YA, int YB, int YC,
	double alpha, double beta);
	
void GPU_add(cublasHandle_t handle, double *A, double *B, double *C,
	int lda, int ldb, int ldc,
	int XA, int YA,
	double alpha, double beta);

void strassen(cublasHandle_t handle, double *A, double *B, double *C,
	int lda, int ldb, int ldc,
	int XA, int XB, int XC,
	int YA, int YB, int YC,
	int depth);

void GPU_strassen(cublasHandle_t handle,double *A, double *B, double *C,
	int lda, int ldb, int ldc,
	int XA, int XB, int XC,
	int YA, int YB, int YC,
	int depth);

void GPU_MUL_TEST(double *A, double *B, double *C, int x, int y, int z);
	


#endif
