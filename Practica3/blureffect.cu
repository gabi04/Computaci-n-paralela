#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

//Namespaces.
using namespace std;
using namespace cv;


//Hallar promedio y retornarlo.
__device__ int cal_intensity(const int* image, int x, int y, int width, int height, int k){

	int ic, ir, fc, fr, n;
	x-(k/2)+1<0 ? ic = 0 : ic = x-(k/2);
	y-(k/2)+1<0 ? ir = 0 : ir = y-(k/2);
	x+(k/2)+1>width ? fc = width : fc = x+(k/2)+1;
	y+(k/2)+1>height ? fr = height : fr = y+(k/2)+1;

	int red = 0, green = 0, blue = 0;
	for(int i=ic; i<fc; i++){
    		for(int j=ir; j<fr; j++){
			n = image[j+i*height];
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
__global__ void blur_thread(const int* d_image, const int width, const int height, const int kernel, const int total_threads, int* d_blur){
	
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int ir = id * ( height / total_threads );
	int fr = (id + 1) * ( height / total_threads );

	if(id < height){
		for(int i=0; i<width; i++){
			for(int j=ir; j<fr; j++){
				d_blur[j+i*height] = cal_intensity(d_image, i, j, width, height, kernel);
			}
		}
	}
}


//Main.
int main(int argc, char** argv){

	//Variables.
	char* image_name;
	Mat image, blur_image;
	int kernel_size, num_threads, num_blocks;
		
	//Recibir argumentos.
	image_name = argv[1];
	kernel_size = atoi(argv[2]);
	num_threads = atoi(argv[3]);
	
	if(argc != 4){
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
	int width = image.cols;
	int height = image.rows;
	blur_image = image.clone();
	cudaError_t err = cudaSuccess;

	//Malloc host
	int numElements = width*height;
	size_t size = numElements * sizeof(int);
	int *h_image = (int *)malloc(size);
	int *h_blur = (int *)malloc(size); 

	//Imagen a un vector 3D
	int aux = 0;
	for(int i=0; i<width; i++){
		for(int j=0; j<height; j++){
			h_image[aux] = image.at<Vec3b>(j,i)[0];
			h_image[aux] += image.at<Vec3b>(j,i)[1] * 1000;
			h_image[aux] += image.at<Vec3b>(j,i)[2] * 1000000;
			aux++;
		}
	}

	//Malloc devise
	//Imagen
	int *d_image = NULL;
	err = cudaMalloc((void **)&d_image, size);
	if(err != cudaSuccess){
		cout<<"Error separando espacio imagen normal en GPU "<<cudaGetErrorString(err)<<endl;
		return -1;
	}
	//Clon
	int *d_blur = NULL;
	err = cudaMalloc((void **)&d_blur, size);
	if(err != cudaSuccess){
		cout<<"Error separando espacio imagen difuminada en GPU "<<cudaGetErrorString(err)<<endl;
		return -1;
	}

	//MemoryCopy
	//Imagen
	err = cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Error copiando datos a GPU "<<cudaGetErrorString(err)<<endl;
		return -1;
	}

	//Lanzar GPU
	int blocksPerGrid = (height + num_threads - 1) / num_threads;	
	blur_thread<<<blocksPerGrid, num_threads>>>(d_image, width, height, kernel_size, height, d_blur);
	err = cudaGetLastError();
	if (err != cudaSuccess){
		cout<<"Fallo al lanzar Kerndel de GPU "<<cudaGetErrorString(err)<<endl;
		return -1;
	}
	
	//Copiar de GPU a CPU
	cout<<"Copiando datos desde la GPU a CPU."<<endl;
    	err = cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout<<"Error copiando desde GPU a CPU "<<cudaGetErrorString(err)<<endl;
		return -1;
	}

	//Escribir imagen difuminada.
	aux = 0;
	for(int i=0; i<width; i++){
		for(int j=0; j<height; j++){
			blur_image.at<Vec3b>(j,i)[0] = (unsigned char)((h_blur[aux]) % 1000);
			blur_image.at<Vec3b>(j,i)[1] = (unsigned char)((h_blur[aux]/1000) % 1000);
			blur_image.at<Vec3b>(j,i)[2] = (unsigned char)((h_blur[aux]/1000000) % 1000);
			aux++;
		}
	}
	imwrite("blur_image.jpg", blur_image);
	

	//Libear espacio
	err = cudaFree(d_image);
	if (err != cudaSuccess){
	        cout<<"Error liberando memoria de imagen normal "<<cudaGetErrorString(err)<<endl;
		return -1;
    	}

	err = cudaFree(d_blur);
	if (err != cudaSuccess){
	        cout<<"Error liberando memoria de imagen difuminada "<<cudaGetErrorString(err)<<endl;
		return -1;
    	}

	free(h_image);
	free(h_blur);

	return 0;
}
