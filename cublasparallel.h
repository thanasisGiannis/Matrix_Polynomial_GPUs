#ifndef CUBLASPARALLEL_H
#define CUBLASPARALLEL_H
#include "omp.h"
#include <cublas_v2.h>

#define ERROR 1
#define OK    0


extern cublasHandle_t *cublasHandler; /* This variable is a pointer of resource handlers.
								 Each one for every device of the system. Is initialized at cublasParallelInit() function
								 and finalized at cublasParallelDestroy() function
							   */

extern int deviceCount; /* Number of devices the system has
					  this variable is initialized at cublasParallelInit() function
					  and finalized at cublasParallelDestroy() function
				    */

extern int cublasParallelInit(); /* Function for resources Initialization */
extern int cublasParallelDestroy(); /* Function for resources Finalization */


#endif
