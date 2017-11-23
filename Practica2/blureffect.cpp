#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

//Namespaces.
using namespace std;
using namespace cv;

//Imagenes.
Mat image, blur_image;

//Variables.
char* image_name;
int kernel_size, num_threads, width, height;

//Hallar promedio y retornarlo.
int cal_intensity(int x, int y, int k, int c){
	int ic, ir, fc, fr, intensity = 0;
	x-(k/2)+1<0 ? ic = 0 : ic = x-(k/2);
	y-(k/2)+1<0 ? ir = 0 : ir = y-(k/2);
	x+(k/2)+1>height ? fc = height : fc = x+(k/2)+1;
	y+(k/2)+1>width ? fr = width : fr = y+(k/2)+1;

	for(int i=ic; i<fc; i++)
    		for(int j=ir; j<fr; j++)
			intensity += image.at<Vec3b>(i,j)[c];

	return intensity/(k*k);
}

//Funcion de cada hilo.
void blur_thread(int ir, int fr){
	for(int i=ir; i<fr; i++)
		for(int j=0; j<width; j++)
			for(int k=0; k<3; k++)
				blur_image.at<Vec3b>(i,j)[k] = cal_intensity(i,j,kernel_size,k);

}

//Main.
int main(int argc, char** argv){
	
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

	//thread m_threads[num_threads];
	omp_set_num_threads(num_threads);
	width = image.cols;
	height = image.rows;
	blur_image = image.clone();

	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		blur_thread(ID*(height/num_threads), (ID+1)*(height/num_threads));
	}


	/*
	//Lanzar hilos dividiendo la imagen verticalmente.
	for(int i=0; i<num_threads; i++){
		m_threads[i] = thread(blur_thread, i*(height/num_threads), (i+1)*(height/num_threads));
	}

	//Esperar por hilos.
	for(int i=0; i<num_threads; i++){
		m_threads[i].join();
	}
	*/

	//Escribir imagen difuminada.
	imwrite("blur_image.jpg", blur_image);
	return 0;
}
