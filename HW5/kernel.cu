#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void mandelKernel(float upperX, float upperY, float lowerX, float lowerY, int* img, float stepX, float stepY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    float width = upperX - lowerX;

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.x;
    
    float x = lowerX + row * stepX;
    float y = lowerY + col * stepY;

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
    img[row*width+col] = i;

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int THREAD =  256;

    //int bx = (upperX-lowerX+THREAD-1)/THREAD;
    //int by = (upperY-lowerY+THREAD-1)/THREAD;

    //dim3 threads(THREAD, THREAD);
    //dim3 blocks(bx, by);

    int *h_img;
    int *d_img;
    int bytes = (upperX-lowerX)*(upperY-lowerY);

    cudaMallocHost(h_img, bytes);
    cudaHostAlloc( (void**)&h_img, bytes ,cudaHostAllocDefault);
    cudaMemcpyDeviceToHost(h_img, img, cudaMemcpyHostToHost);
    cudaMalloc(d_img, bytes);

    cudaMemcpy(h_img, img, bytes, cudaMemcpyHostToHost);
    cudaMemcpy(d_img, img, bytes, cudaMemcpyHostToDevice);

    dim3 threads(resX, resY);
    int blocks = 1;
    mandelKernel<<<blocks, threads>>>(float upperX, float upperY, float lowerX, float lowerY, int* img, float stepX, float stepY, int maxIterations);
    cudaMemcpy(img, d_img, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(h_img);
}
