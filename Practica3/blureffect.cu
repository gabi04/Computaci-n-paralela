#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

//Namespaces.
using namespace std;
using namespace cv;

//Hallar promedio y retornarlo.
//__global__ int cal_intensity(int x, int y, int k, int c){
__global__ int cal_intensity(int* image, int x, int y, int width, int height, int k){

	int ic, ir, fc, fr, n, intesity;
	x-(k/2)+1<0 ? ic = 0 : ic = x-(k/2);
	y-(k/2)+1<0 ? ir = 0 : ir = y-(k/2);
	x+(k/2)+1>width ? fc = width : fc = x+(k/2)+1;
	y+(k/2)+1>height ? fr = height : fr = y+(k/2)+1;

	int red = 0, green = 0, blue = 0;
	for(int i=ic; i<fc; i++){
    		for(int j=ir; j<fr; j++){
			n = image[i+j*height];
			blue += (n % 1000);
			green += (n/1000) % 1000;
			red += (n/1000000) % 1000;
		}
	}

	blue = blue / (k*k);
	green = green / (k*k);
	red = red / (k*k);
	return (red*1000000)+(green*1000)+blue;
}

//Funcion de cada hilo.
__global__ void blur_thread(const int* d_image, const int width, const int height, const int kernel, const int total_threads, int d_blur){

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int ir = id * ( height / total_threads );
	int fr = (id + 1) * ( height / total_threads );

	for(int i=0; i<width; i++){
		for(int j=ir; j<fr; j++){
			d_blur[i+j*height] = cal_intesity(d_image, i, j, width, height, kernel);
		}
	}
}

//Main.
int main(int argc, char** argv){

	//Variables.
	char* image_name;
	Mat image, blur_image;
	int kernel_size, num_threads, num_blocks, width, height;
		
	//Recibir argumentos.
	image_name = argv[1];
	kernel_size = atoi(argv[2]);
	num_threads = atoi(argv[3]);
	num_blocks = atoi(argv[4]);

	if(argc != 5){
		cout<<"Numero incorrecto de argumentos.\n";
		return -1;
	}

	//Leer imagen
	image = imread(image_name);
	if(!image.data){
		cout<<"Imagen no reconocida.\n";
		return -1;
	}

	//Inicializar variables
	width = image.cols;
	height = image.rows;
	blur_image = image.clone();
	cudaError_t err = cudaSuccess;


	//Malloc host
	int numElements = width*height*3;
	size_t size = numElements*sizeof(int);
	int *h_image = (int *)malloc(size);

	//Imagen a un vector 3D
	int aux = 0;
	for(int i=0; i<width; i++){
		for(int j=0; j<height; j++){
			h_image[aux] = image.at<Vec3b>(i,j)[0];
			h_image[aux] += image.at<Vec3b>(i,j)[1] * 1000;
			h_image[aux] += image.at<Vec3b>(i,j)[2] * 1000000;
			aux++;
		}
	}
	
	//Malloc devise
	//Width
	int d_width = NULL;
	err = cudaMalloc((void *)d_width, sizeof(int));
	if(err != cudaSuccess){
		cout<<"Error separando espacio en GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}
	//Height
	int d_height = NULL;
	err = cudaMalloc((void *)d_height, sizeof(int));
	if(err != cudaSuccess){
		cout<<"Error separando espacio en GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}
	//Kernel
	int d_kernel = NULL;
	err = cudaMalloc((void *)d_kernel, sizeof(int));
	if(err != cudaSuccess){
		cout<<"Error separando espacio en GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}
	//Array
	int *d_image = NULL;
	err = cudaMalloc((void **)&d_image, size);
	if(err != cudaSuccess){
		cout<<"Error separando espacio en GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}

	//MemoryCopy
	//Imagen
	err = cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Error copiando datos a GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}
	//Width
	err = cudaMemcpy(d_width, width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Error copiando datos a GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}
	//Kernel
	err = cudaMemcpy(d_kernel, kernel_size, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Error copiando datos a GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}
	//Height
	err = cudaMemcpy(d_height, height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Error copiando datos a GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}

	//Lanzar GPU
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	cout<<"Kernel de CUDA lanzado con "<<num_blocks<<" bloques y "<<num_threads<<" hilos."<<endl;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
	err = cudaGetLastError();
	if (err != cudaSuccess){
		cout<<"Fallo al lanzar Kerndel de GPU "<<cudaGetErrorString(err))<<endl;
		return -1;
	}


	//Escribir imagen difuminada.
	imwrite("blur_image.jpg", blur_image);
	return 0;
}
