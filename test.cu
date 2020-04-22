#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include "matrixfunctions.h"
#include <cblas.h>	
/* ******** NAIVE FUNCTION FOR TESTING PURPOSE *********/




void scalMatrix(double *B,int incB,double *A,int incA,int rows,int cols,double scalar){

	memcpy(B,A,rows*cols*sizeof(double));
	cblas_dscal(rows*cols,scalar,B,1);
	
}


void addDiag(double *B,int incB,double *A,int incA,int rows,int cols,double scalar){

	memcpy(B,A,rows*cols*sizeof(double));
	int i,j;
	for(i=0;i<rows;i++)
			B[i*incB+i] = scalar+B[i*incA+i];
}


void naiveMatrixAdd(double *C,int incC,double *A,int incA,double *B,int incB,int rows,int cols){

	int i,j;
	for(i=0;i<rows;i++)
		for(j=0;j<cols;j++)
			C[i*incC+j] = A[i*incA+j] + B[i*incB+j];

}


void mul(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    double alpha, double beta) {
//  	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,XA,YA,XB,alpha,A,lda,B,ldb,beta,C,ldc);

	}

void add(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int YA,
    double alpha, double beta) {

//  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &beta, B, ldb, C, ldc);
	
	int i,j;
	for(i=0;i<XA;i++)
		for(j=0;j<YA;j++)
			C[i*ldc+j] = alpha*A[i*lda+j] + beta*B[i*ldb+j];
	
}


void serialStrassen(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    int depth) {

  int XA2 = XA / 2;
  int XB2 = XB / 2;
  int XC2 = XC / 2;
  
  int YA2 = YA / 2;
  int YB2 = YB / 2;
  int YC2 = YC / 2;

  //double *W_1, *W_2;
  int lw1 = (XA2 > XC2 ? XA2 : XC2);
  int lw2 = XB2;
  //cudaMalloc((void **)&W_1, lw1 * YA2 * sizeof(double));
  double* W_1 = (double*)malloc(lw1 * YA2 * sizeof(double)); 

// cudaMalloc((void **)&W_2, lw2 * YB2 * sizeof(double));
  double* W_2 = (double*)malloc(lw2 * YB2 * sizeof(double));

  if( W_1 == NULL ) printf("Error\n"); 
  if( W_2 == NULL ) printf("Error2\n");
  
  int dXA = XA2;
  int dYA = YA2 * lda;
  int dXB = XB2;
  
  int dYB = YB2 * ldb;
  int dXC = XC2;
  int dYC = YC2 * ldc;

  double *A11, *A12, *A21, *A22;
  double *B11, *B12, *B21, *B22;
  double *C11, *C12, *C21, *C22;
  
  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;
  
  B11 = B;
  B12 = B + dXB;
  B21 = B + dYB;
  B22 = B + dXB + dYB;
  
  C11 = C;
  C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;

  /* cutoff criteria */
  bool stop = false;
  
#if 0
  int cutoff = 2048;
  float mm = cutoff / XB2;
  float nn = cutoff / YA2;
  float kk = cutoff / XA2;
  if ((mm + nn + kk) >= 3) {
      stop = true;
  }
#endif

  if (depth <= 1 || stop) {
    add( A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    add( B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    mul( W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = W_1 * W_2
    add( A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0,  1.0); // W_1 = A21 + A22
    add( B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    mul( W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C22 = W_1 * W_2
    add( W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    add( B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    mul( W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = W_1 * W_2
    add( A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    mul( W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C12 = W_1 * B22
    add( C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C22 + C12
    mul( A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // W_1= A11 * B11
    add( W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1 + C11
    add( C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C11 + C12
    add( C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = C11 + C21
    add( W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    mul( A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = A22 * W_2
    add( C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    add( C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C22 = C11 + C22
    mul( A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = A12 * B21
    add( W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1+ C11
  }
  else {
    add( A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    add( B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    serialStrassen( W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0,  1.0); // W_1 = A21 + A22
    add( B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    serialStrassen( W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    add( B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    serialStrassen( W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    serialStrassen( W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C22 + C12
    serialStrassen( A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1 + C11
    add( C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C11 + C12
    add( C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = C11 + C21
    add( W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    serialStrassen( A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    add( C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C22 = C11 + C22
    serialStrassen( A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    add( W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1+ C11
  }
  free(W_1);
  free(W_2);

  /* dynamic peeling fix-up */
  int pxa = XA % 2;
  int pya = YA % 2;
  int pxb = XB % 2;
  int pyb = YB % 2;
  int pxc = XC % 2;
  int pyc = YC % 2;
  
  int nxa = XA - pxa;
  int nya = YA - pya;
  int nxb = XB - pxb;
  int nyb = YB - pyb;
  int nxc = XC - pxc;
  int nyc = YC - pyc;

  double *a12, *a21;
  double *b12, *b21;
  double *c12, *c21;
  int dxa = nxa;
  int dya = nya * lda;
  int dxb = nxb;
  int dyb = nyb * ldb;
  int dxc = nxc;
  int dyc = nyc * ldc;
  
  a12 = A + dxa;
  a21 = A + dya;
  // a22 = A + dxa + dya;
  b12 = B + dxb;
  b21 = B + dyb;
  // b22 = B + dxb + dyb;
  c12 = C + dxc;
  c21 = C + dyc;
  // c22 = C + dxc + dyc;

  /* 
    A11 = nxa x nya
    a12 = pxa x nya
    a21 = nxa x pya
    a22 = pxa x pya
   */
  mul( a21, B11, c21, lda, ldb, ldc, nxa,  XB,  XC, pya, nyb, pyc, 1.0, 0.0);
  mul( A11, b12, c12, lda, ldb, ldc, nxa, pxb, pxc,  YA, nyb,  YC, 1.0, 0.0);
  mul( a12, b21, C11, lda, ldb, ldc, pxa,  XB,  XC,  YA, pyb,  YC, 1.0, 1.0);

 }

void naiveMatrixMul(double* C,int incC,double* A,int incA,double *B,int incB,int m,int k,int n){


	double *tmp = (double*)malloc(m*n*sizeof(double));
	memset(tmp,0,m*n*sizeof(double));
	//serialStrassen(A,B,tmp,incA,incB,m,m,k,m,k,n,n,1); 
	///serialStrassen(A,B,C,incA,incB,incC,m,k,m,k,n,n,1); 
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,incA,B,incB,0.0,tmp,n);

	//printf("error strassen\n");
	memcpy(C,tmp,m*n*sizeof(double));
	free(tmp);

	
   
}


void naivePolComp(double *B,int incB,double *A,int incA,int rows,int cols,double *coef,int coefNum){

	double *tmpMatrix  = (double*)malloc(rows*cols*sizeof(double));
	double *tmpMatrix2 = (double*)malloc(rows*cols*sizeof(double));
	double *A_2        = (double*)malloc(rows*cols*sizeof(double));

	naiveMatrixMul(A_2,cols,A,incA,A,incA,rows,cols,cols) ;
	//matrixMul(A_2,cols,A,incA,A,incA,rows,cols,cols); /* A_2 = A*A */
/*
	double e=0,normA_2=0;
	for(int i =0;i<rows*cols;i++){
		e = abs(A_2[i] - tmpMatrix[i])*abs(A_2[i] - tmpMatrix[i]);
		normA_2 = A_2[i]*A_2[i];
	}

	printf("sfalma= %e\n",(double)e/normA_2);
	return;
*/

	int loopStart;
	if( (coefNum % 2) == 0 ) { 
		
		/* if polynomial order is even compute the aI + bX */
	     scalMatrix(B,incB,A,incA,rows,cols,coef[coefNum-1]);
		addDiag(B,incB,B,incB,rows,cols,coef[coefNum-2]);
		loopStart=coefNum-3;

	}else{
		/* if polynomial order is odd compute the aI */

		memset(tmpMatrix,0,rows*cols*sizeof(double));
		addDiag(B,incB,tmpMatrix,cols,rows,cols,coef[coefNum-1]);
		loopStart=coefNum-2;


	}

	for(int i =loopStart;i>=0;i=i-2){
		naiveMatrixMul(B,incB,A_2,cols,B,incB,rows,cols,cols) ;
		//matrixMul(B,incB,A_2,cols,B,incB,rows,cols,cols); /*B = X_2*B	*/	
		
		scalMatrix(tmpMatrix,cols,A,incA,rows,cols,coef[i]);
		addDiag(tmpMatrix,cols,tmpMatrix,cols,rows,cols,coef[i-1]);
		naiveMatrixAdd(B,incB,B,incB,tmpMatrix,cols,rows,cols);
	
	}

	free(tmpMatrix);
	free(A_2);
}

/* **************************************************** */

int main(int argc, char **argv) {

/* this program is called like that ./test -d degree -r rows */

	srand(time(0));
    double time_spent;
	unsigned long start,end;
	struct timeval  tv1,tv2;	
	int opt,rows=10,cols=10,deg=2;
    extern char   *optarg;
	
	while ( (opt=getopt(argc,argv,"r:d:h"))!= EOF) {
        switch (opt) {
            case 'r': rows=atoi(optarg);
			          cols=rows;
                      break;
            case 'd': deg = atoi(optarg);
                      break;
            default:  break;
        }
    }    
	
	double *x = (double*)malloc(rows*cols*sizeof(double));
	double *xout = (double*)malloc(rows*cols*sizeof(double));
	double *coef = (double*)malloc(deg*sizeof(double));

	double *xtest = (double*)malloc(rows*cols*sizeof(double)); /*****/




	int i,j;

		for(i=0;i<rows;i++)
    {
		for(j=0;j<rows;j++)
	    {
			x[i*rows+j] = (double)(rand()%10)/12;
		}
		
	}
	for(i=0;i<rows*cols;i++) xout[i] = 1;//x1[i];
	for(i=0;i<deg;i++) 
	{
		coef[i] = (double)(rand()%10)/12;
	}

	printf("GPUs polynomial computation...\n");

	cublasParallelInit();
	
	gettimeofday(&tv1, NULL);
	initializeZero(xout,cols,rows,cols);
	matrixPol(xout,cols,x,cols,rows,cols,coef,deg);
	gettimeofday(&tv2, NULL);
	
	cublasParallelDestroy();
	
	start = (unsigned long)(tv1.tv_usec + tv1.tv_sec * 1000000);
	end = (unsigned long)(tv2.tv_usec + tv2.tv_sec * 1000000);

	time_spent=(double)((end - start) / 1000000.0);

	clock_t begin2,end2;
	double time_spent2;

     printf("Done in GPUs\nNaive method computation in CPU...\n");

	cublasParallelInit();

	begin2 = clock();
	naivePolComp(xtest,cols,x,cols,rows,cols,coef,deg);  /****/
	end2 = clock();

	cublasParallelDestroy();
		
	printf("Done in CPU\n");
	time_spent2 = (double)(end2-begin2)/CLOCKS_PER_SEC;

	printf("Execution time GPUs:%lfs CPU:%lf \n",time_spent,time_spent2);
	/*****/ 
	double resDif=0;  
	double resX  =0;
	double resT  =0;
	for(i=0;i<rows;i++)
    {
		for(j=0;j<rows;j++)
	    {
	        resDif += (xout[i]-xtest[i])*(xout[i]-xtest[i]);
		    resX   += xout[i]*xout[i];
		    resT   += xtest[i]*xtest[i];
		}
		
		
	}
	
	printf("||Xgpu-Xcpu||_2 %e\n",(double)sqrt(resDif/resX));	
	printf("||Xgpu||_2 %e\n",(double)sqrt(resX));
	printf("||Xcpu||_@ %e\n",(double)sqrt(resT));
	
	free(xtest);  /*****/
	free(xout);
	free(x);
	free(coef);

 
}
