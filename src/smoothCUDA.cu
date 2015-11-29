#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define BLUE 0
#define GREEN 1
#define RED 2

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void loadShared(uchar shared[5][260][3], uchar * data, int i, int j, int x, int height, int step, int channels)
{
    int k;
    for(k = 0; k < 5; k++)
    {
        int pos = i - 2 + k;
        if ( pos > 0 && pos < height)
        {
            shared[k][x][0] = data[pos * step + j * channels];
            shared[k][x][1] = data[pos * step + j * channels + 1];
            shared[k][x][2] = data[pos * step + j * channels + 2];
        }
    }
}

__global__ void gpuSmooth(uchar * target, uchar * data, int width, int height, int step, int channels)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    int x = threadIdx.x + 2;
    int value[3];
    __shared__ uchar mem[5][260][3];
    int total;
    int k, l, m;
    
    // Spill treatment
    if((i > 0) && (i < height) && (j > 0) && (j < width))
    {
        // Load values to shared memory
        loadShared(mem, data, i, j, x, height, step, channels);
        if (x == 2)
        {
            if (j > 1) 
            {
               loadShared(mem, data, i, j-1, x-1, height, step, channels);
               loadShared(mem, data, i, j-2, x-2, height, step, channels);
            }
        }
        else if (x == 257)
        {
            if (j + 1 < width)
            {
                loadShared(mem, data, i, j+1, x+1, height, step, channels);
            }
            if (j + 2 < width)
            {
                loadShared(mem, data, i, j+2, x+2, height, step, channels);
            }
        }
    }

    __syncthreads();    

    if((i > 0) && (i < height) && (j > 0) && (j < width))
    {
        total = value[0] = value[1] = value[2] = 0;
        for(k = 0; k < 5; k++)
            if ((i - 2 + k > 0) && (i - 2 + k < height))
                for(l = x - 2, m = j - 2; l < x+3; l++, m++)
                    if((m > 0) && (m < width))
                    {
                        value[0] += mem[k][l][0];
                        value[1] += mem[k][l][1];
                        value[2] += mem[k][l][2];
                        ++total;
                    }
        target[i * step + j * channels] = value[0] / total; 
        target[i * step + j * channels + 1] = value[1] / total;
        target[i * step + j * channels + 2] = value[2] / total;
    }

}

int main(int argc, char *argv[])
{
    // original image
    IplImage* img = 0;

    if(argc<2){
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }

    // load an image  
    img=cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img){
        printf("Could not load image file: %s\n",argv[1]);
        exit(0);
    }

    int image_size = img->height*img->widthStep;

    uchar * gpu_data, *gpu_target;
    gpuErrchk(cudaSetDevice(1));
    gpuErrchk(cudaMalloc(&gpu_data, image_size));
    gpuErrchk(cudaMalloc(&gpu_target, image_size));

    gpuErrchk(cudaMemcpy(gpu_data, img->imageData, image_size, cudaMemcpyHostToDevice));

    dim3 grid(img->height, (img->width / 256) + (img->width % 256 != 0), 1);

    gpuSmooth<<<grid, 256>>>(gpu_target, gpu_data, img->width, img->height, img->widthStep, img->nChannels);
    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(img->imageData, gpu_target, image_size, cudaMemcpyDeviceToHost));

    cvSaveImage("result/result.jpg", img, 0);
    // release the image
    cvReleaseImage(&img);
    cudaFree(gpu_data);
    cudaFree(gpu_target);
    return 0;
}
