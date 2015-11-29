#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define BLUE 0
#define GREEN 1
#define RED 2

void applySmooth(IplImage*, IplImage*);
int calculatePixel(uchar*, int, int , int , int , int , int , int );

__global__ void gpuSmooth(uchar * target, uchar * data, int width, int height, int step, int channels)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int value[3];
    int total;
    int k, l;


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

    gpuSmooth<<<img->height, img->width>>>(gpu_target, gpu_data, img->width, img->height, img->widthStep, img->nChannels);

    cudaMemcpy(img->imageData, gpu_target, image_size, cudaMemcpyDeviceToHost);

    cvSaveImage("result/result.jpg", img, 0);
    // release the image
    cvReleaseImage(&img);
    cudaFree(gpu_data);
    cudaFree(gpu_target);
    return 0;
}

void applySmooth(IplImage* img, IplImage* img_result){
    uchar *data   = NULL;
    uchar *data_result = NULL;

    data = (uchar *)img->imageData;

    // get the data from the copied image
    // we have to work with this one because we dont want to mess up the
    // original file
    data_result = (uchar *)img_result->imageData;

    // get the image data
    int height    = img->height;
    int width     = img->width;
    int step      = img->widthStep;
    int channels  = img->nChannels;
    
    // Pixels from the border do have less pixels aronund than the others, we
    // have consider this value while calculating the newValue
    int i, j;

    //  For each pixel in the image
    for(i=0;i<height;i++){
        for(j=0;j<width;j++){
            // New image values
            data_result[i*step+j*channels+RED]= calculatePixel(data, i, j, height, width, step, channels, RED);
            data_result[i*step+j*channels+GREEN]= calculatePixel(data, i, j, height, width, step, channels, GREEN);
            data_result[i*step+j*channels+BLUE]= calculatePixel(data, i, j, height, width, step, channels, BLUE);

        }
    } 
}

int calculatePixel(uchar* data, int i, int j, int height, int width, int step, int channels, int color){

    int value = 0;
    int newValue = 0;
    int k,l;
    // For each pixel in the matrix 5x5 around the current pixel
    for(k = i-2; k < i + 3; k++){
        for(l = j-2; l < j+3; l++){

            // Check if the pixel exists (it may be outside the grid)
            if((k > 0) && (k < height) && (l > 0) && (l < width)){

                // Adds all the pixel values of the 5x5 matrix
                newValue = newValue + data[k*step+l*channels+color];
                //newRedValue = newRedValue + data[k*step+l*channels+RED];
                //newGreenValue = newGreenValue + data[k*step+l*channels+GREEN];
                //newBlueValue = newBlueValue + data[k*step+l*channels+BLUE];
                value++;
            }                           
        }
    }
    return newValue/value;
}
