__kernel void convolution(
	__global float *input,
	__global float *output,
	int *imageWidth,
	int *imageHeight,
	float* filter,
	int *filterSize) 
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

	int width = *imageWidth;

	int halffiltersize = (*filterSize)/2;

	int sum = 0;
	
//sdfejfsefsefh	

//	for(int i=-halffiltersize; i<=halffilter; i++){
//		for(int j=-halffiltersize;j<=halffiltersize;j++) {
//			if((col+j>=0) && (col+j <width) && (row+i>=0) && *(row+i<height)){
//				sum += input[(row+i)*width+col+j] * filter[(i+halffiltersize)*(*filterSize)+(j+halffiltersize)];
//			}
//		}
//	}

//	output[row*(width-2)+col] = sum;
}
