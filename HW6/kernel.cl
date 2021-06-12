__kernel void convolution(
	__global float* input,
	__global float* output,
	int imageWidth,
	int imageHeight,
	__global float* filter,
	int filterWidth) 
{

    const int col = get_global_id(0);
    const int row = get_global_id(1);


//	int outputWidth = imageWidth-filterWidth+1;
	int sum = 0;
/*
	for(int i=0;i<filterWidth;i++) {
		for(int j=0;j<filterWidth;j++) {
			sum += input[(row+i)*imageWidth+(col+j)]*filter[i*filterWidth+j];
		}
	}

	output[row*(outputWidth)+col] = sum;
*/
	int halffilterSize = filterWidth/2;	

	int i = row;
	int j = col;
	int k, l;

    for (k = -halffilterSize; k <= halffilterSize; k++)
	{
		for (l = -halffilterSize; l <= halffilterSize; l++)
		{
			if (i + k >= 0 && i + k < imageHeight &&
					j + l >= 0 && j + l < imageWidth)
			{
					sum += input[(i + k) * imageWidth + j + l] *
						   filter[(k + halffilterSize) * filterWidth +
								  l + halffilterSize];
			}
		}
	}
    output[i * imageWidth + j] = sum;

}
