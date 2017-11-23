#include <omp.h>
#include <stdio.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv){
	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		cout<<"Hello "<<ID<<".\n";
		cout<<"World "<<ID<<".\n";
	}
	return 0;
}
