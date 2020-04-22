#include "matrixfunctions.h"
#include <stdio.h>

int updateDiag(double *,int,double *,int,double ,int,int);
int updateMatrix(double *,int,double*,int,double,int,int);
void matrixPol(double *,int ,double *,int,int,int,double*,int);
void initializeZero(double*,int,int,int);
void matrixAdd(double*,int,double*,int,double*,int,int,int);
void naiveMatrixPol(double*,int,double*,int,int,int,double*,int);

__global__
void setValue(double* dev_vec,int value , int size){
 
     /* device kernel that takes for input a vector and an integer and set all cells of vector to this integer */

	 int tid = (gridDim.y*blockIdx.x+blockIdx.y)*blockDim.x*blockDim.y+blockDim.y*threadIdx.x+threadIdx.y;
      if (tid < size) {
           dev_vec[tid] = value;
       }


}



void matrixAdd(double* C,int incC,double* A,int incA,double* B,int incB,int rows,int cols){

	/*	This function implements the matrix addition in multiple gpus.

		Every thread (device) is responsible for the outcome of a number of rows of matrix C.
		eg.

		| a11 a12 a13 .... a1N | <--- Gpu 1
		| a21 a22 a23 .... a2N |
		| a31 a32 a33 .... a3N | <--- Gpu 2 
		| a41 a42 a43 .... a4N |
		|  .   .   .  ....  .  |       etc
		| aM1 aM2 aM3 .... aMN | 

	*/

	int tmpDeviceCount = deviceCount;
	if(rows<deviceCount) deviceCount =1;
	omp_set_num_threads(deviceCount);

	#pragma omp parallel
	{
		double *dev_A,*dev_B,*dev_C;
		double one = 1.0;
		int numThread = omp_get_thread_num();
		cudaSetDevice(numThread);
		int sizeRows = rows / deviceCount;
		int offsetA = numThread*sizeRows*incA;
		int offsetB = numThread*sizeRows*incB;
		int offsetC = numThread*sizeRows*incC;
	
		if(numThread == deviceCount -1 ) sizeRows += rows%deviceCount;	
	
		cudaMalloc((void**)&dev_A,sizeRows*cols*sizeof(double));
		cudaMalloc((void**)&dev_B,sizeRows*cols*sizeof(double));
		cudaMalloc((void**)&dev_C,sizeRows*cols*sizeof(double));

		cublasSetMatrix(cols,sizeRows,sizeof(double),&A[offsetA],incA,dev_A,cols);
		cublasSetMatrix(cols,sizeRows,sizeof(double),&B[offsetB],incB,dev_B,cols);

		cublasDgeam(cublasHandler[numThread],CUBLAS_OP_N,CUBLAS_OP_N,cols,sizeRows,&one,dev_A,cols,&one,dev_B,cols,dev_C,cols);

		cublasGetMatrix(cols,sizeRows,sizeof(double),dev_C,cols,&C[offsetC],incC);
		
	//printf("1\n");
		cudaFree(dev_A);
		cudaFree(dev_B);
		cudaFree(dev_C);
	}
	
	deviceCount = tmpDeviceCount;
}


int updateDiag(double *outputMatrix,int incOutput,double *inputMatrix,int incInput,double alpha,int rows,int cols){

	/* This function implements 
		B = aI + A

		The idea goes like this

		set a vector v to ones
		multiply v with a and add it to the diag of the input matrix
		last step is implemented using cublasDaxpy (y = y +a*x where y,x are vectors and a is scalar)

		Every thread (device) is responsible for a block of output vector. 

		eg.

			| v1 | <-- Gpu 1
			| v2 |
			| v3 | <-- Gpu 2
			| v4 |
			| .  |      etc
			| vN |
	*/	
	int tmpDeviceCount = deviceCount;
	if(rows < deviceCount) deviceCount =1;
	
	int *errorStatus = (int*)malloc(deviceCount*sizeof(int));
	
	omp_set_num_threads(deviceCount);
	#pragma omp parallel
	{
		int numThread = omp_get_thread_num();
		cudaSetDevice(numThread);	

		int sizeDiag  = rows / deviceCount;
		int offsetIn  = numThread*sizeDiag*(incInput+1);
		int offsetOut = numThread*sizeDiag*(incOutput+1);
		if(numThread == deviceCount -1 ) sizeDiag += rows % deviceCount;

		double *dev_input,*dev_output;
		cudaMalloc((void**)&dev_output,sizeDiag*sizeof(double));
		cudaMalloc((void**)&dev_input,sizeDiag*sizeof(double));

		if(cublasSetVector(sizeDiag,sizeof(double),&inputMatrix[offsetIn],incInput+1,dev_input,1)!=CUBLAS_STATUS_SUCCESS)
				errorStatus[numThread] = ERROR;

		int grid = ceil((sizeDiag)/1024); 	
		if(grid<=0)grid=1;

		int BLOCK_SIZE=512;
		int number_of_blocks = ((sizeDiag) + BLOCK_SIZE - 1) / BLOCK_SIZE;
          dim3 gridDim(number_of_blocks, 1);
          dim3 blockDim(BLOCK_SIZE, 1);
		
		setValue<<<gridDim, blockDim>>>(dev_output,1,sizeDiag);
	
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) 
    		printf("Error: %s\n", cudaGetErrorString(err));

		cublasDaxpy(cublasHandler[numThread],sizeDiag,&alpha,dev_output,1,dev_input,1);
		
	
		if(cublasGetVector(sizeDiag,sizeof(double),dev_input,1,&outputMatrix[offsetOut],incOutput+1)!=CUBLAS_STATUS_SUCCESS)
				errorStatus[numThread] = ERROR;		

		cudaFree(dev_input);
		cudaFree(dev_output);
	}

	deviceCount = tmpDeviceCount;	
	int index;
	for(index=0;index<deviceCount;index++){
			
		if(errorStatus[index] != OK ) return ERROR;

	}

	return OK;
}


int updateMatrix(double *outMatrix,int incOut,double *inMatrix,int incIn,double alpha,int rows,int cols){

	/* This function implements the update of a matrix

		B = a*A;
	
	  Core of this function is the cublasDscal that does this exact operation.
	  The blocking of the input matrix goes like this

	  Every thread (device) gets a number of rows that is responsible for the outcome
		
		eg 

		| a11 a12 a13 .... a1N | <--- Gpu 1
		| a21 a22 a23 .... a2N |
		| a31 a32 a33 .... a3N | <--- Gpu 2 
		| a41 a42 a43 .... a4N |
		|  .   .   .  ....  .  |       etc
		| aM1 aM2 aM3 .... aMN | 

	*/

	int tmpDeviceCount = deviceCount;
	if(rows < deviceCount) deviceCount =1;
	
	int *errorStatus = (int*)malloc(deviceCount*sizeof(int));
	omp_set_num_threads(deviceCount);
	#pragma omp parallel
	{
		int numThread = omp_get_thread_num();
		cudaSetDevice(numThread);
		int sizeRows = rows / deviceCount;
		int offsetIn   = numThread*sizeRows*incIn;
		int offsetOut  = numThread*sizeRows*incOut;

		if(numThread == deviceCount-1) sizeRows += rows%deviceCount;
		double *dev_inMatrix;
		
		cudaMalloc((void**)&dev_inMatrix,sizeRows*cols*sizeof(double));	
		
		cublasSetMatrix(cols,sizeRows,sizeof(double),&inMatrix[offsetIn],incIn,dev_inMatrix,cols);

		cublasDscal(cublasHandler[numThread],sizeRows*cols,&alpha,dev_inMatrix,1);
		
		cublasGetMatrix(cols,sizeRows,sizeof(double),dev_inMatrix,cols,(void*)&outMatrix[offsetOut],incOut);
		cudaFree(dev_inMatrix);
	}

	deviceCount = tmpDeviceCount;
	int index;
	for(index=0;index<deviceCount;index++){
		if(errorStatus[index] != OK ) return ERROR;

	}
	return OK;
}






int divisor(int number){

	/* Just a nothing-to-say-about-function passing by */

    int i;
    for (i = number / 2; i >= 1; i--)
    {
        if (number % i == 0)
        {
            break;
        }
    }
    return i;
}






void matrixMul(double* C,int ldc,double* A,int lda,double* B,int ldb,int m,int k,int n){
/*	input A,B
	output C
*/	

	/* This function implements the block matrix multiplication 
	   Function divisor given a number a returns primitive number b so that b*c = a.
	   So, given a we know b and we find c. By those two numbers (b and c )we define the number of rows and columns 
	   inside of a block.

	   After that every thread (device) takes an outcome-block and the needed income blocks and 
	   eventually calling the GPU_strassen function for doing the multiplication.

	  
	*/
				
	int numA,numB;
	int stepA,stepB;


	numA = divisor(deviceCount);
	if(numA==0)
	   numA=1;
	numB = deviceCount/numA;
	 
	stepA = m/numA; // Number of rows per block
	stepB = n/numB; // Number of columns per block

	int i,j; // for counters

	int tmpDeviceCount=deviceCount;
	if(deviceCount > m || deviceCount > n) deviceCount = 1;
	omp_set_num_threads(deviceCount);
	#pragma omp parallel for collapse(2)
	for(i=0;i<numA;i++){
		for(j=0;j<numB;j++){
		
		unsigned int numThread = omp_get_thread_num();
		if(cudaSetDevice(numThread)!=cudaSuccess){
			printf("ERROR");
			//exit;
		}
		

		if(i==numA-1 && j==numB-1){
			
			GPU_strassen(cublasHandler[numThread],&A[i*stepA*lda],&B[j*stepB],&C[i*stepA*ldc+j*stepB],lda,ldb,ldc,stepA+m % numA,k,stepA+m % numA,k,stepB+n % numB,stepB+n % numB,1);

		}else if(i==numA-1 && j!=numB-1){

			GPU_strassen(cublasHandler[numThread],&A[i*stepA*lda],&B[j*stepB],&C[i*stepA*ldc+j*stepB],lda,ldb,ldc,stepA+m%numA,k,m%numA+stepA,k,stepB,stepB,1);

		}else if(i!=numA-1 && j==numB-1){

			GPU_strassen(cublasHandler[numThread],&A[i*stepA*lda],&B[j*stepB],&C[i*stepA*ldc+j*stepB],lda,ldb,ldc,stepA,k,stepA,k,stepB+n%numB,stepB+n%numB,1);

		} else {

			GPU_strassen(cublasHandler[numThread],&A[i*stepA*lda],&B[j*stepB],&C[i*stepA*ldc+j*stepB],lda,ldb,ldc,stepA,k,stepA,k,stepB,stepB,1);

		}

	
		}
	}

	deviceCount = tmpDeviceCount;


}


void initializeZero(double *X,int incX,int rows,int cols){
	
	/* This function given an input matrix X, returns matrix X with all values set to 0.
	   Core of this function if the function cudaMemset()
	*/

	int tmpDeviceCount = deviceCount;
	if(rows<deviceCount) deviceCount = 1;
	//deviceCount=1;	
	omp_set_num_threads(deviceCount);
	
	#pragma omp parallel
	{
		int numThread = omp_get_thread_num();
		cudaSetDevice(numThread);
		int sizeRows = rows / deviceCount;
		int offsetX = numThread*sizeRows*incX;	
		if(numThread == deviceCount-1) sizeRows += rows % deviceCount;
		double *dev_X;
		cudaMalloc((void**)&dev_X,sizeRows*cols*sizeof(double));
		cudaMemset(dev_X,0,sizeRows*cols*sizeof(double));
	
		cublasGetMatrix(cols,sizeRows,sizeof(double),dev_X,cols,&X[offsetX],incX);
		cudaFree(dev_X);

	}

	deviceCount = tmpDeviceCount;
	
}


void matrixPol(double *B,int incB,double *A,int incA,int rows,int cols,double* coef,int coefNum){
/**** B = f(A,coef); ****/
	
	/* This function is responsible for the computational flow of the polynomial
	   
	   Inside this function 2 additional matrices are been used

	   Step1: Calculate A^2
	   Step2: Caclulate first or first two orders of the polynomial,by doing that we are sure that inside the loop are even orders
	   Step3: Compute rest of the polynomial inside the loop

	*/

	double *tmpMatrix = (double*)malloc(rows*cols*sizeof(double));
	double *A_2 	   = (double*)malloc(rows*cols*sizeof(double));

	matrixMul(A_2,cols,A,incA,A,incA,rows,cols,cols); /* A_2 = A*A */

	int loopStart;
	if( (coefNum % 2) == 0 ) { 
		
		/* if polynomial order is even compute the aI + bX */
	
		updateMatrix(B,incB,A,incA,coef[coefNum-1],rows,cols);
		updateDiag(B,incB,B,incB,coef[coefNum-2],rows,cols);
		loopStart=coefNum-3;
	}else{
		/* if polynomial order is odd compute the aI */

		initializeZero(tmpMatrix,cols,rows,cols);
		updateDiag(B,incB,tmpMatrix,cols,coef[coefNum-1],rows,cols);
		loopStart=coefNum-2;
	}

	int i;
	for(i =loopStart;i>=0;i=i-2){
	
		/*Rest of the polynomial orders are computed here */
		matrixMul(B,incB,A_2,cols,B,incB,rows,cols,cols); /*B = X_2*B	*/	
		updateMatrix(tmpMatrix,cols,A,incA,coef[i],rows,cols);/* a*X */
		updateDiag(tmpMatrix,cols,tmpMatrix,cols,coef[i-1],rows,cols); /* b*I+a*X */
		matrixAdd(B,incB,B,incB,tmpMatrix,cols,rows,cols);	/* B =B + b*I+a*X	*/
		
		
	}

	free(tmpMatrix);
	free(A_2);

}











