#include <stdlib.h>
#include <omp.h>
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define BLUE 0
#define GREEN 1
#define RED 2

void applySmooth(IplImage*, IplImage*, int start, int end);
int calculatePixel(uchar*, int, int , int , int , int , int , int );

int main(int argc, char *argv[])
{
    int numtasks, rank, rc, dest, source, count, tag=1;
    int number_of_processes;
    char inmsg[10], outmsg[10]; 

    // original image
    IplImage* img = 0;

    // result image
    IplImage* img_result = 0;

    uchar *img_local_data = 0;


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
    // Copy the image to apply the smooth algorithm 
    img_result = cvCloneImage(img);

    printf("height: %d width: %d\n", img->height, img->width);
    // Initialize MPI and get important variable.
    MPI_Status Stat;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);


    // Image size
    int imageSize = img_result->height*img_result->widthStep;
    printf("Image size: %d\n", imageSize);

    


    // Calculates the amount of work of each process
    int workload = img_result->height / number_of_processes;

    int rec_size = workload * img_result->widthStep;

    if(rank == 0){

        // Array that contains the final imageData
        printf("Calculating from 0 to %d\n", workload);
        char *result = (char*)malloc(sizeof(char)*imageSize);
        int total_work = workload * number_of_processes;

        // The first process always calculates the first part of the matrix
        applySmooth(img, img_result, 0, workload);
      
        // Collect all the processed data
        MPI_Gather(img_result->imageData, rec_size, MPI_CHAR, result, rec_size, MPI_CHAR, 0, MPI_COMM_WORLD);


        // add the new imageData array to plot the image
        img_result->imageData = result;

        if (total_work < img_result->height)
        {
            applySmooth(img, img_result, total_work, img_result->height);
            printf ("Root calculated %d extra rows, from %d to %d\n", img_result->height - total_work, total_work, img_result->height);
        }
        cvSaveImage("result/result.jpg", img_result, 0);
        cvReleaseImage(&img);
   }
    else {
      
        // Each process knows where to start
        int start = rank*workload;
        printf("Calculating from %d to %d\n", start, start+workload);
        printf ("Passing %d bytes %d %d\n", imageSize/number_of_processes, rec_size, workload * img_result->widthStep);

        applySmooth(img, img_result, start, start+workload);

        // The offset is important to avoid any problems while building the
        // image


        MPI_Gather(img_result->imageData + rec_size * rank, rec_size, MPI_CHAR, NULL , rec_size, MPI_CHAR, 0, MPI_COMM_WORLD); 
    }


    MPI_Finalize();

    return 0;
}

void applySmooth(IplImage* img, IplImage* img_result, int start, int end){
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
    // values for each pixel
    int newRedValue = 0;
    int newGreenValue = 0;
    int newBlueValue = 0;

    // Pixels from the border do have less pixels aronund than the others, we
    // have consider this value while calculating the newValue
    int value = 0;
    int i, j, k, l;

    #pragma omp parallel for private(i, j)
    //  For each pixel in the image
    for(i=start;i<end;i++){
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
                value++;
            }                           
        }
    }
    return newValue/value;
}
