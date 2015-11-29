#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define BLUE 0
#define GREEN 1
#define RED 2

__global__ void gpuSmooth(uchar * target, uchar * data, int width, int height, int step, int channels)
{
    int i; i = blockIdx.x;
    int j; j = threadIdx.x + blockIdx.y * blockDim.x;
    int value[3];
    int total;
    int k, l;

    if((i < 0) || (i > height) || (j < 0) || (j > width))
        return;

    total = value[0] = value[1] = value[2] = 0;

    for(k = i-2; k < i + 3; k++)
        for(l = j-2; l < j+3; l++)
            if((k > 0) && (k < height) && (l > 0) && (l < width))
            {
                value[0] += data[k * step + l * channels];
                value[1] += data[k * step + l * channels + 1];
                value[2] += data[k * step + l * channels + 2];
                ++total;
            }

    target[i * step + j * channels] = value[0] / total; 
    target[i * step + j * channels + 1] = value[1] / total;
    target[i * step + j * channels + 2] = value[2] / total;

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

    cudaMalloc(&gpu_data, image_size);
    cudaMalloc(&gpu_target, image_size);

    cudaMemcpy(gpu_data, img->imageData, image_size, cudaMemcpyHostToDevice);

    dim3 grid(img->height, (img->width / 256) + (img->width % 256 != 0), 1);

    gpuSmooth<<<grid, 256>>>(gpu_target, gpu_data, img->width, img->height, img->widthStep, img->nChannels);

    cudaMemcpy(img->imageData, gpu_target, image_size, cudaMemcpyDeviceToHost);

    cvSaveImage("result/result.jpg", img, 0);
    // release the image
    cvReleaseImage(&img);
    cudaFree(gpu_data);
    cudaFree(gpu_target);
    return 0;
}
