#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 220
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include<math.h>
#include <CL/cl.h>
#include<string.h>

typedef struct WAV_RIFF {
    /* chunk "riff" */
    char ChunkID[4];   /* "RIFF" */
    /* sub-chunk-size */
    uint32_t ChunkSize; /* 36 + Subchunk2Size */
    /* sub-chunk-data */
    char Format[4];    /* "WAVE" */
} RIFF_t;

typedef struct WAV_FMT {
    /* sub-chunk "fmt" */
    char Subchunk1ID[4];   /* "fmt " */
    /* sub-chunk-size */
    uint32_t Subchunk1Size; /* 16 for PCM */
    /* sub-chunk-data */
    uint16_t AudioFormat;   /* PCM = 1*/
    uint16_t NumChannels;   /* Mono = 1, Stereo = 2, etc. */
    uint32_t SampleRate;    /* 8000, 44100, etc. */
    uint32_t ByteRate;  /* = SampleRate * NumChannels * BitsPerSample/8 */
    uint16_t BlockAlign;    /* = NumChannels * BitsPerSample/8 */
    uint16_t BitsPerSample; /* 8bits, 16bits, etc. */
} FMT_t;

typedef struct WAV_data {
    /* sub-chunk "data" */
    char Subchunk2ID[4];   /* "data" */
    /* sub-chunk-size */
    uint32_t Subchunk2Size; /* data size */
    /* sub-chunk-data */
//    Data_block_t block;
} Data_t;

//typedef struct WAV_data_block {
//} Data_block_t;

typedef struct WAV_fotmat {
    RIFF_t riff;
    FMT_t fmt;
    Data_t data;
} Wav;


char* seconds_to_time(float raw_seconds) {
	char* hms;
	int hours, hours_residue, minutes, seconds, milliseconds;
	hms = (char*)malloc(100);

	sprintf(hms, "%f", raw_seconds);

	hours = (int)raw_seconds / 3600;
	hours_residue = (int)raw_seconds % 3600;
	minutes = hours_residue / 60;
	seconds = hours_residue % 60;
	milliseconds = 0;

	// get the decimal part of raw_seconds to get milliseconds
	char* pos;
	pos = strchr(hms, '.');
	int ipos = (int)(pos - hms);
	char decimalpart[15];
	memset(decimalpart, ' ', sizeof(decimalpart));
	strncpy(decimalpart, &hms[ipos + 1], 3);
	milliseconds = atoi(decimalpart);


	sprintf(hms, "%d:%d:%d.%d", hours, minutes, seconds, milliseconds);
	return hms;
}


void write_little_endian(unsigned int word, int num_bytes, FILE* wav_file)
{
	unsigned buf;
	while (num_bytes > 0)
	{
		buf = word & 0xff;
		fwrite(&buf, 1, 1, wav_file);
		num_bytes--;
		word >>= 8;
	}
}


void write_wav(const char* filename, Wav* wav, unsigned int* data_block)
{
	FILE* wav_file;

	FMT_t fmt;
	Data_t data;

	fmt = wav->fmt;
	data = wav->data;

	wav_file = fopen(filename, "wb");
	if (!wav_file) return; /* make sure it opened */


	/* write RIFF header */
	fwrite("RIFF", 1, 4, wav_file);

	write_little_endian(36 + data.Subchunk2Size, 4, wav_file);

	fwrite("WAVE", 1, 4, wav_file);

	/* write fmt subchunk */
	fwrite("fmt ", 1, 4, wav_file);

	/* SubChunk1Size is 16 */
	write_little_endian(fmt.Subchunk1Size, 4, wav_file);
	write_little_endian(fmt.AudioFormat, 2, wav_file);    /* PCM is format 1 */
	write_little_endian(fmt.NumChannels, 2, wav_file);
	write_little_endian(fmt.SampleRate, 4, wav_file);
	write_little_endian(fmt.ByteRate, 4, wav_file);
	write_little_endian(fmt.BlockAlign, 2, wav_file);  /* block align */
	write_little_endian(fmt.BitsPerSample, 2, wav_file);  /* bits/sample */

	/* write data subchunk */
	fwrite("data", 1, 4, wav_file);
	write_little_endian(data.Subchunk2Size, 4, wav_file);
	fwrite(data_block, 1, data.Subchunk2Size, wav_file);
	printf("Proslo\n");
	fclose(wav_file);
}
Wav* open_wav(FILE* fp, float** data_b)
{
	Wav* wav = (Wav*)malloc(sizeof(Wav));
	RIFF_t riff;
	FMT_t fmt;
	Data_t data;

	fread(wav, 1, sizeof(Wav), fp);


	riff = wav->riff;
	fmt = wav->fmt;
	data = wav->data;

	float* data_block = NULL;
	data_block = (float*)malloc(data.Subchunk2Size);
	if (!data_block) {
		printf("Neuspjesno alociranje prostora\n");
		return 0;
	}

	fread(data_block, sizeof(float), data.Subchunk2Size / sizeof(float), fp);

	*data_b = data_block;


	printf("ChunkID \t%c%c%c%c\n", riff.ChunkID[0], riff.ChunkID[1], riff.ChunkID[2], riff.ChunkID[3]);
	printf("ChunkSize \t%d\n", riff.ChunkSize);
	printf("Format \t\t%c%c%c%c\n", riff.Format[0], riff.Format[1], riff.Format[2], riff.Format[3]);

	printf("\n");

	printf("Subchunk1ID \t%c%c%c%c\n", fmt.Subchunk1ID[0], fmt.Subchunk1ID[1], fmt.Subchunk1ID[2], fmt.Subchunk1ID[3]);
	printf("Subchunk1Size \t%d\n", fmt.Subchunk1Size);
	printf("AudioFormat \t%d\n", fmt.AudioFormat);
	printf("NumChannels \t%d\n", fmt.NumChannels);
	printf("SampleRate \t%d\n", fmt.SampleRate);
	printf("ByteRate \t%d\n", fmt.ByteRate);
	printf("BlockAlign \t%d\n", fmt.BlockAlign);
	printf("BitsPerSample \t%d\n", fmt.BitsPerSample);

	printf("\n");

	printf("blockID \t%c%c%c%c\n", data.Subchunk2ID[0], data.Subchunk2ID[1], data.Subchunk2ID[2], data.Subchunk2ID[3]);

	printf("\n");

	printf("blockSize \t%d\n", data.Subchunk2Size);

	printf("duration \t%s\n", seconds_to_time(data.Subchunk2Size / fmt.ByteRate));
	return wav;
}


float* make_sincs(Wav* wav, int* ret_size, unsigned int frequency) {
	Wav w = *wav;
	int num_elements=40000;
	int size = num_elements / sizeof(float);
	*ret_size = size;
	int zero = size / 2;
	float* sinc = (float*)calloc(num_elements,1);
	float amplitude;
	for (int i = 0; i < size; i++) {
		amplitude = 1 / ((float)(i - zero) / w.fmt.SampleRate * 3.1415 * (50000-frequency));
		sinc[i] = amplitude * sin(2 * 3.1415 * frequency * (i - zero) / w.fmt.SampleRate);

	}
	sinc[zero] = (sinc[zero - 1] + sinc[zero + 1]) / 2;
	return sinc;
}


static char* readKernelSource(const char* filename)
{
	char* kernelSource = NULL;
	long length;
	FILE* f = fopen(filename, "r");
	if (f)
	{
		fseek(f, 0, SEEK_END);
		length = ftell(f);
		fseek(f, 0, SEEK_SET);
		kernelSource = (char*)calloc(length, sizeof(char));
		if (kernelSource)
			fread(kernelSource, 1, length, f);
		fclose(f);
	}
	return kernelSource;
}

void runAudioExample(Wav* wav, float* data, float* sinc, float* result, int data_size, int sinc_size)
{
	// Length of vectors
	unsigned int n = wav->data.Subchunk2Size / sizeof(float);
	int d_size = data_size, s_size = sinc_size;

	// Host output vector

	// Device output buffer
	cl_mem cl_result,
		cl_data,
		cl_sinc;

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel

	// Size, in bytes, of each vector
	size_t bytes = n * sizeof(float);

	// Allocate memory for each vector on host

	size_t globalSize, localSize;
	cl_int err;

	// Number of work items in each local work group
	localSize = 80;

	// Number of total work items - localSize must be devisor
	globalSize = (size_t)ceil(n / (float)localSize) * localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	char* kernelSource = readKernelSource("Convolution.cl");

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);

	// Build the program executable 
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err)
	{
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char* log = (char*)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);

		free(log);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "convolve", &err);

	// Create the input and output arrays in device memory for our calculation
	cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	cl_data = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size * sizeof(float), NULL, NULL);
	cl_sinc = clCreateBuffer(context, CL_MEM_READ_ONLY, sinc_size * sizeof(float), NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, cl_data, CL_TRUE, 0, data_size * sizeof(float), data, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, cl_sinc, CL_TRUE, 0, sinc_size * sizeof(float), sinc, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	unsigned int result_len = data_size + sinc_size;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_result);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_data);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_sinc);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &d_size);
	err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &s_size);
	err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &result_len);
	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, bytes, result, 0, NULL, NULL);

	// release OpenCL resources
	clReleaseMemObject(cl_result);
	clReleaseMemObject(cl_data);
	clReleaseMemObject(cl_sinc);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	free(kernelSource);
}

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

float* convolve(float h[], float x[], float* y, int lenH, int lenX)
{
	int nconv = lenH + lenX - 1;
	int i, j, h_start, x_start, x_end;

#pragma omp parallel for
	for (i = 0; i < nconv; i++)
	{
		x_start = MAX(0, i - lenH + 1);
		x_end = MIN(i + 1, lenX);
		h_start = MIN(i, lenH - 1);
		for (j = x_start; j < x_end; j++)
		{
			y[i] += h[h_start--] * x[j];
		}
	}
	printf("CPU zavrsio\n");
	return y;
}

int main(int argc, char* argv[])
{
	int FREQUENCY=1000;
	FILE* fp = NULL;
    if(argc < 2){
        return 1;
	}

	fp = fopen(argv[1], "rb");
	if (!fp) {
		printf("can't open audio file\n");
		exit(1);
	}
    if(argc == 3){
        FREQUENCY = atoi(argv[2]);
    }
	float* data_block;
	float* sinc_signal;
	float* result;
	int data_size, sinc_size;
	Wav* wav;
	printf("PODACI O AUDIO ZAPISU\n");
	printf("=====================\n");
	wav = open_wav(fp, &data_block);
	printf("=====================\n");
	sinc_signal = make_sincs(wav, &sinc_size, FREQUENCY);
	printf("\nSINC kreiran\n");

	data_size = wav->data.Subchunk2Size / sizeof(float);
	wav->data.Subchunk2Size = (data_size + sinc_size) * sizeof(float);
	wav->riff.ChunkSize = 36 + wav->data.Subchunk2Size;

	result = (float*)calloc(wav->data.Subchunk2Size,1);

	/*--------------------------------konvolucija----------------------------------*/
	double start, end;
	start = omp_get_wtime();
	if (wav->data.Subchunk2Size > 2000000) {
		printf("Koristen OpenCL\n");
		runAudioExample(wav, data_block, sinc_signal, result, data_size, sinc_size);
	}
	else {
		printf("Koristen OpeMP\n");
		convolve(sinc_signal, data_block, result, sinc_size, data_size);
	}
	end = omp_get_wtime();
	printf("Vrijeme izvrsavanja: %s\n", seconds_to_time(end - start));
	/*------------------------------------------------------------------------------*/

	printf("Zavrsena konvolucija\n");

	write_wav("rezultat_konvolucije.wav", wav, (unsigned int*)result);
	printf("Upisano u fajl\n");
	free(data_block);
	free(sinc_signal);
	free(result);
	return 0;
}