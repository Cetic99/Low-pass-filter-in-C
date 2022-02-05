#define PI 3.141592653589793238462643383279502884
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

__kernel void convolve(__global float* result,
							const __global float* orig_signal,
							const __global float* sinc_signal,
							const unsigned int orig_len,
							const unsigned int sinc_len,
							const unsigned int result_len)
{
	int nconv = result_len;
	int j,h_start,x_start,x_end;

    int id = get_global_id(0);
	
	x_start = MAX(0,id-sinc_len+1);
    x_end   = MIN(id+1,orig_len);
    h_start = MIN(id,sinc_len-1);
    for(j=x_start; j<x_end; j++)
    {
      result[id] += sinc_signal[h_start--]*orig_signal[j];
    }
}