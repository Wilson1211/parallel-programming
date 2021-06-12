#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
	int halffilterWidth = filterWidth/2;
/*
// create context
	cl_int ciErrNum;

	context = clCreateContextFromType (0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);

// get platform ID
	cl_int err_num;

	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);

	ciErrNum = clGetDeviceIDs (0, CL_DEVICE_TYPE_GPU, 1, &device,  cl_uint *num_devices);
*/

	printf("Enter hostFE\n");
	cl_int ciErrNum, error_code;

	cl_command_queue myqueue ;
	myqueue = clCreateCommandQueue( *context, *device, 0, &ciErrNum);

	printf("width: %d, height: %d\n", imageWidth, imageHeight);

// create buffer; create to 
	int ip_mem_size = imageHeight*imageWidth*sizeof(float);
	//int op_mem_size = (imageHeight-filterWidth+1)*(imageWidth-filterWidth+1)*sizeof(int);
	int op_mem_size = imageHeight*imageWidth*sizeof(float);
	

	cl_mem d_ip = clCreateBuffer( *context, CL_MEM_READ_ONLY, ip_mem_size, NULL, &ciErrNum);
	cl_mem d_op = clCreateBuffer( *context, CL_MEM_WRITE_ONLY, op_mem_size, NULL, &ciErrNum);
	cl_mem d_filter = clCreateBuffer( *context, CL_MEM_READ_ONLY, filterSize*sizeof(float), NULL, &ciErrNum);

// copy data to device 
	ciErrNum = clEnqueueWriteBuffer ( myqueue , d_ip, CL_TRUE, 0, ip_mem_size, (void *)inputImage, 0, NULL,  NULL); 
	ciErrNum = clEnqueueWriteBuffer ( myqueue , d_filter, CL_TRUE, 0, filterSize*sizeof(float), (void *)filter, 0, NULL,  NULL); 

// crate kernel
	cl_kernel mykernel = clCreateKernel ( *program , "convolution" , error_code);

// Set Arguments 
	clSetKernelArg(mykernel, 0, sizeof(cl_mem), (float *)&d_ip); 	
	clSetKernelArg(mykernel, 1, sizeof(cl_mem), (float *)&d_op); 	
	clSetKernelArg(mykernel, 2, sizeof(int), (void *)&imageWidth); 	
	clSetKernelArg(mykernel, 3, sizeof(int), (void *)&imageHeight); 	
	clSetKernelArg(mykernel, 4, sizeof(cl_mem), (float *)&d_filter); 	
	clSetKernelArg(mykernel, 5, sizeof(int), (void *)&filterWidth); 	

//Set local and global workgroup sizes
	size_t localws[2] = {10,10} ; 
//	size_t globalws[2] = {imageWidth-filterWidth+1, imageHeight-filterWidth+1};//Assume divisible by 16
	size_t globalws[2] = {imageWidth, imageHeight};//Assume divisible by 16

// execute kernel
	clEnqueueNDRangeKernel( myqueue , mykernel, 2, 0, globalws, localws, 0, NULL, NULL);

// copy results from device back to host
	clEnqueueReadBuffer( myqueue, d_op,  CL_TRUE, 0, op_mem_size,  (void *) outputImage,  NULL, NULL, NULL);
	printf("hostFE finish\n");

	clReleaseMemObject(d_ip);
    clReleaseMemObject(d_op);
    clReleaseMemObject(d_filter);
	
    clReleaseKernel(mykernel);
    clReleaseCommandQueue(myqueue);


}
