#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void mandelKernel(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float x = lowerX + col * stepX;
    float y = lowerY + row * stepY;

    int i;
    float z_re = x, z_im = y;
    for (i = 0; i < maxIterations; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f){
            break;
        }

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    img[row*resX+col] = i;
    //printf("row: %d, col: %d, i: %d\n", row, col, i);

}

void test(int *img, int X, int Y) {

    FILE *fp;
    int i, j;

    fp = fopen("log.txt", "w");
    for(i=0;i<Y;i++){
	for(j=0;j<X;j++){
	    fprintf(fp, "%d ", img[i*X+j]);
	}
	fprintf(fp, "\n");
    }
    fclose(fp);

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{

    //int bx = (upperX-lowerX+THREAD-1)/THREAD;
    //int by = (upperY-lowerY+THREAD-1)/THREAD;

    //dim3 threads(THREAD, THREAD);
    //dim3 blocks(bx, by);

    int *d_img, *h_img;
//    size_t bytes = (upperX-lowerX)*(upperY-lowerY)*sizeof(int);
    size_t bytes = resX*resY*sizeof(int);
    h_img = (int*) malloc(bytes);


    cudaMalloc((void **)&d_img, bytes);

    //cudaMemcpy(h_img, img, bytes, cudaMemcpyHostToHost);
    cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice);

    int THREAD = 16;
    
    printf("resX: %d, resY: %d, maxIterations: %d\n", resX, resY, maxIterations);
    int bx = (resX+THREAD-1)/THREAD;
    int by = (resY+THREAD-1)/THREAD;

    dim3 threads(THREAD, THREAD);
    dim3 blocks(bx, by);
    mandelKernel<<<blocks, threads>>>(upperX, upperY, lowerX, lowerY, d_img, resX, resY, maxIterations);
    cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost);

//    cudaMemcpy(img, h_img, bytes, cudaMemcpyHostToHost);
    memcpy(img, h_img, bytes);
//    printf("h_img[0]: %d\n", h_img[0]);
//    printf("sizeof img: %x\n", sizeof(img)/sizeof(*img));
//    test(img, resX, resY);
    test(h_img, resX, resY);


    cudaFree(d_img);
    free(h_img);
}
