#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

void test(int *img, int X, int Y) {

    FILE *fp;
    int i, j;

    fp = fopen("log.txt", "w");
    for(i=0;i<Y;i++){
	for(j=0;j<X;j++){
//	    printf("j:%d ", j);
	    fprintf(fp, "%d ", img[i*X+j]);
	}
//	printf("\n");
	fprintf(fp, "\n");
    }
    fclose(fp);

}

__global__ void mandelKernel(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, size_t pitch, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

//    int* row_matrix = img+row*pitch;
    
    float x = lowerX + col * stepX;
    float y = lowerY + row * stepY;

    int i;
    float z_re = x, z_im = y;
    for (i = 0; i < maxIterations; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
        break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    float* index = (float*)((char*)img+row*pitch);
    index[col] = i;
//    row_matrix[col] = i;
//    img[row*pitch+col] = i;
//    img[row*resX+col] = i;

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    //int bx = (upperX-lowerX+THREAD-1)/THREAD;
    //int by = (upperY-lowerY+THREAD-1)/THREAD;

    //dim3 threads(THREAD, THREAD);
    //dim3 blocks(bx, by);

    int *h_img;
    int *d_img;

//    int bytes = sizeof(float)*resX*resY;
    int bytes = resX*resY;

//    cudaMallocHost(h_img, bytes);
//    cudaMemcpy(h_img, img, bytes, cudaMemcpyHostToHost);
//    cudaMemcpyDeviceToHost(h_img, img, cudaMemcpyHostToHost);
  //  cudaMalloc(d_img, bytes);

    size_t pitch;
    cudaMallocPitch((void**)&d_img, (size_t *)&pitch, sizeof(int)*resX, resY);

    printf("pitch: %lx\n", pitch);
//cudaMemcpy2D(   host_memory, /* dest */
//		  100*sizeof(float)   /*no pitch on host*/,
//                myArray,     /* src */
//		  pitch/*CUDA pitch*/,
//		  100*sizeof(float)/*width in bytes*/, 
//		  100/*heigth*/, 
//		  cudaMemcpyDeviceToHost);

//    cudaHostAlloc( (void**)&h_img, sizeof(int)*bytes ,cudaHostAllocDefault);
    h_img = (int*)malloc(sizeof(int)*bytes);
    cudaMemcpy2D(d_img, pitch, h_img, sizeof(int)*resX, sizeof(int)*resX, resY,cudaMemcpyHostToDevice);


    int THREAD = 16;
    int bx = (resX+THREAD-1)/THREAD;
    int by = (resY+THREAD-1)/THREAD;
    	
    dim3 threads(THREAD, THREAD);
    dim3 blocks(bx, by) ;
    mandelKernel<<<blocks, threads>>>(upperX, upperY, lowerX, lowerY, d_img, resX, resY, pitch, maxIterations);
    //cudaMemcpy(img, d_img, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_img, sizeof(int)*resX, d_img, pitch, sizeof(int)*resX, resY,cudaMemcpyDeviceToHost);

    cudaMemcpy(img, h_img, sizeof(int)*bytes, cudaMemcpyHostToHost);

    test(h_img, resX, resY);
    cudaFree(d_img);
//    cudaFreeHost(h_img);
    free(h_img);
}
