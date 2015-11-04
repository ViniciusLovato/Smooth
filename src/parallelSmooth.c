#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define BLUE 0
#define GREEN 1
#define RED 2

#define N 10

double average = 0;

void applySmooth(IplImage*, uchar*, int start, int end);

int main(int argc, char *argv[])
{
    int numtasks, rank, rc, dest, source, count, tag=1;
    int number_of_processes;
    char inmsg[10], outmsg[10]; 

    // original image
    IplImage* img = 0;
    uchar *img_local_data = 0;


    if(argc<2){
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }

    // load an image  
    
    // Initialize MPI and get important variable.
    MPI_Status Stat;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    img=cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img){
        printf("Could not load image file: %s\n",argv[1]);
        exit(0);
    }

    // Image size
    int imageSize = img->height*img->widthStep;
    //printf("Image size: %d\n", imageSize);
    fflush(stdout);
    

    // Calculates the amount of work of each process
    int workload = img->height / number_of_processes;

    int rec_size = workload * img->widthStep;

    int i;

    for(i = 0; i < N; i++){
        if(rank == 0){

            struct timespec start, finish;
            double time_spent;

            clock_gettime(CLOCK_MONOTONIC, &start);

            // Array that contains the final imageData
            //printf("Calculating from 0 to %d\n", workload);
            char *result = malloc(rec_size);
            char *extra = 0;
            int total_work = workload * number_of_processes;

            // The first process always calculates the first part of the matrix
            applySmooth(img, result, 0, workload);

            if (total_work < img->height)
            {
                extra = malloc((img->height - total_work) * img->widthStep);

                applySmooth(img, extra, total_work, img->height);

                //printf ("Root calculated %d extra rows, from %d to %d\n", img->height - total_work, total_work, img->height);
                memcpy(img->imageData + rec_size * number_of_processes, extra, (img->height - total_work) * img->widthStep);
            }

            // Collect all the processed data
            MPI_Gather(result, rec_size, MPI_CHAR, img->imageData, rec_size, MPI_CHAR, 0, MPI_COMM_WORLD);

            clock_gettime(CLOCK_MONOTONIC, &finish);

            time_spent = (finish.tv_sec - start.tv_sec);
            time_spent += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

            printf("Execution# %d - Time spent %lf\n", i, time_spent);

            average = average + time_spent;

            // cvSaveImage("result/result.jpg", img, 0);
            // cvReleaseImage(&img);
            free(result);
            free(extra);
        }
        else {

            // Each process knows where to start
            int start = rank*workload;
            char *  result = malloc(rec_size);
            //printf("Calculating from %d to %d\n", start, start+workload);
            //printf ("Passing %d bytes %d %d\n", imageSize/number_of_processes, rec_size, workload * img->widthStep);

            applySmooth(img, result, start, start+workload);

            // The offset is important to avoid any problems while building the
            // image
            MPI_Gather(result, rec_size, MPI_CHAR, NULL , rec_size, MPI_CHAR, 0, MPI_COMM_WORLD); 
            free(result);
        }
    
    }

    if(rank == 0)
    {

        average = average/N;
        printf("Average %lf\n", average);
    }
    MPI_Finalize();
    return 0;
}


void applySmooth(IplImage* img, uchar* img_result, int start, int end){
    uchar *data   = NULL;
    uchar *data_result = NULL;

    data = (uchar *)img->imageData;

    // get the data from the copied image
    // we have to work with this one because we dont want to mess up the
    // original file
    data_result = (uchar *)img_result;

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
    int value = 0, sum[3];
    int i, j, k, l;
    int load = end - start;


#pragma omp parallel for private(i, j, k, l, value, sum)
    //  For each pixel in the image
    for(i=0;i<load;i++){
        for(j=0;j<width;j++){
            int sum[3];
            value = sum[RED] = sum[GREEN] = sum[BLUE] = 0;
            for(k = i + start -2; k < i + start + 3; k++){
                for(l = j-2; l < j+3; l++){
                    // Check if the pixel exists (it may be outside the grid)
                    if((k > 0) && (k < height) && (l > 0) && (l < width)){
                        // New image values
                        sum[RED] += data[k*step+l*channels+RED];
                        sum[GREEN] += data[k*step+l*channels+GREEN];
                        sum[BLUE] += data[k*step+l*channels+BLUE];
                        value++;
                    }
                }
            }
            data_result[i*step+j*channels+RED]= sum[RED] / value;
            data_result[i*step+j*channels+GREEN]= sum[GREEN] / value;
            data_result[i*step+j*channels+BLUE]= sum[BLUE] / value;
        }
    } 
}
