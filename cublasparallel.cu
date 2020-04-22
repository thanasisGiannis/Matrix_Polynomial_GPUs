#include "cublasparallel.h"


cublasHandle_t *cublasHandler;
int deviceCount;



int cublasParallelInit(){
	/* Initialization of the resources that are used for the polynomial computation */

	/* Every thread is binded with a different device, so in each thread the resources for 
	   different device are been initialized 
	*/

	int *errorStatus;
	
	cudaGetDeviceCount(&deviceCount);

	omp_set_num_threads(deviceCount);
	
	cublasHandler = (cublasHandle_t*)malloc(deviceCount*sizeof(cublasHandle_t));	
	errorStatus   = (int*)malloc(deviceCount*sizeof(int));


	#pragma omp parallel
	{

		int threadNum = omp_get_thread_num();
		cudaSetDevice(threadNum);
		
		if(cublasCreate(&cublasHandler[threadNum]) != CUBLAS_STATUS_SUCCESS){
				errorStatus[threadNum] = ERROR;
	
		}
		

	}

	int index;
	for(index = 0;index<deviceCount;index++){
			if(errorStatus[index] == ERROR) return ERROR;
	}

	return OK;
}


int cublasParallelDestroy(){

	/* For every thread, a device is binded. 
	   So, in each thread the resources for the device are been finalized.
	*/
	int *errorStatus;
	errorStatus = (int*)malloc(deviceCount*sizeof(int));
	omp_set_num_threads(deviceCount);

	#pragma omp parallel
	{
			int threadNum = omp_get_thread_num();
			cudaSetDevice(threadNum);

			if(cublasDestroy(cublasHandler[threadNum])!= CUBLAS_STATUS_SUCCESS){
				errorStatus[threadNum] = ERROR;

			}

	}

	int index;
	for(index=0;index<deviceCount;index++){

			if(errorStatus[index] != OK) return ERROR;
	}
	
	return OK;
}



















