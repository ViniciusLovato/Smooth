#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define BLUE 0
#define GREEN 1
#define RED 2

int main(int argc, char *argv[])
{
    // original image
    IplImage* img = 0;
    int height,width,step,channels;
    uchar *data;

    // result image
    IplImage* img_result = 0;
    uchar *data_result;
    int height_result,width_result,step_result,channels_result;

    int i,j,k, l;

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

    // get the image data
    height    = img->height;
    width     = img->width;
    step      = img->widthStep;
    channels  = img->nChannels;
    data      = (uchar *)img->imageData;


    // creates a new image
    img_result = cvCloneImage(img);

    // get the data from the copied image
    // we have to work with this one because we dont want to mess up the
    // original file
    data_result      = (uchar *)img_result->imageData;

    // create a window
    cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE); 
    cvMoveWindow("mainWin", 20, 20);

    // values for each pixel
    int newRedValue = 0;
    int newGreenValue = 0;
    int newBlueValue = 0;

    // Pixels from the border do have less pixels aronund than the others, we
    // have consider this value while calculating the newValue
    int value = 0;

    //  For each pixel in the image
    for(i=0;i<height;i++){
        for(j=0;j<width;j++){

            // The new value separate in the 3 channels
            newRedValue = 0;
            newGreenValue = 0;
            newBlueValue = 0;
            value = 0; 

            // For each pixel in the matrix 5x5 around the current pixel
            for(k = i-2; k < i + 3; k++){
                for(l = j-2; l < j+3; l++){
                    
                    // Check if the pixel exists (it may be outside the grid)
                    if((k > 0) && (k < height) && (l > 0) && (l < width)){

                            // Adds all the pixel values of the 5x5 matrix
                            newRedValue = newRedValue + data[k*step+l*channels+RED];
                            newGreenValue = newGreenValue + data[k*step+l*channels+GREEN];
                            newBlueValue = newBlueValue + data[k*step+l*channels+BLUE];
                            value++;
                    }                           
                }
            }
            
            // New image values
            data_result[i*step+j*channels+RED]= newRedValue/value;
            data_result[i*step+j*channels+GREEN]= newGreenValue/value;
            data_result[i*step+j*channels+BLUE]= newBlueValue/value;

        }
    }
    // show the image
    cvShowImage("mainWin", img_result);

    // wait for a key
    cvWaitKey(0);

    // release the image
    cvReleaseImage(&img );
    return 0;
}


