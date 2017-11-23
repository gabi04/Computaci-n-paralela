#include "omp.h"
#include "stdio.h"

#define NUM_THREADS 2
#define PAD 8

static long num_steps = 10000;
double step;

double iter(int n){
	double i, res;
	if (n % 2 == 0)
		i = 1.0;
	else
		i = -1.0;

	res = i * (4.0 / (2.0 * (double)n+1));
	return res;
}

void main(){
	int nthreads = NUM_THREADS;
	double pi, sum[NUM_THREADS][PAD];
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int id = omp_get_num_threads();		
		for(int i = id; i<num_steps; i){			
			sum[id][0] += iter(i);
		}
	}
	
	for(int i=0, pi=0.0; i<nthreads; i++)
		pi += sum[i][0];
	printf("PI: %f \n", pi);

	pi = 0.0;
	for(int i=0; i<1000000; i++){
		pi += iter(i);
	}
	printf("%f \n", pi);
}

