/********************************** HEADING ***********************************
*																			  *
*									Detect.cpp 								  *
*							 C. Bellamy   J. Steiner                          *
*																			  *
******************************************************************************/

/***************************** LOAD DEPENDENCIES *****************************/

//Loads in the class header
#include "Detect.h"

/****************************** BLOB CONSTRUCTOR ******************************
* Name:  Blob
* Param: x - the initial x point of the blob
*        y - the initial y point of the blob
* Notes: class constructor
******************************************************************************/
Blob::Blob(int x, int y)
{
    //Defaults the x and y corner points
    Blob::minX = x; Blob::maxX = x;
    Blob::minY = y; Blob::maxY = y;
}

/**************************** DISTANCE TO FUNCTION ****************************
* Name:  distanceTo
* Param: x - the x point we are checking the distance to
*        y - the y point we are checking the distance to
* Notes: finds the shortest distance to the blob from the x and y point
******************************************************************************/
int Blob::distanceTo(int x, int y)
{
    //Calculates the distance from the blob to the left side
    int distanceToLeft =  pow(Blob::minX - x, 2) + pow(Blob::minY - y, 2);

    //Calculates the distance from the blob to the right side
    int distanceToRight = pow(Blob::maxX - x, 2) + pow(Blob::maxY - y, 2);

    //Returns the closest distance (right or left)
    return std::min(distanceToLeft, distanceToRight);
}

/**************************** ADD PIXEL TO FUNCTION ***************************
* Name:  addPx
* Param: x - the x point we are adding to the blob
*        y - the y point we are adding to the blob
* Notes: adds the x and y point to the blob
******************************************************************************/
void Blob::addPx(int x, int y)
{
    //The new max x is whichever is larger: the current max x or new x
    Blob::maxX = std::max(Blob::maxX, x);

    //The new max y is whichever is larger: the current max y or new y
    Blob::maxY = std::max(Blob::maxY, y);

    //The new min x is whichever is smaller: the current min x or new x
    Blob::minX = std::min(Blob::minX, x);

    //The new min y is whichever is smaller: the current min y or new y
    Blob::minY = std::min(Blob::minY, y);
}

/******************************* GETTER FUNCTIONS *****************************
* Name:  getMaxX, getMinX, getMaxY, getMinY
* Notes: returns the private variables
******************************************************************************/
int Blob::getMaxX() { return Blob::maxX; }
int Blob::getMaxY() { return Blob::maxY; }
int Blob::getMinX() { return Blob::minX; }
int Blob::getMinY() { return Blob::minY; }

Detect::Detect(int threshold)
{
    Detect::blobThreshold = threshold;
}

/****************************** DETECT FUNCTION *******************************
* Name:  detect
* Param: img  - the image we will be searching through
*        type - the type of detection that we will be doing
* Notes: loops through the image and assigns pixels that meet our threshold to
*        a blob object
******************************************************************************/
void Detect::searchFrame(cv::Mat& frame1, cv::Mat& frame2, int color, 
                         int threshold, int type)
{
    //Gets how many channels are in the image
    int channels = frame1.channels();

    //Calculates the amount of rows and columnswe will need to loop through
    int nRows = frame1.rows;
    int nCols = frame1.cols * channels;

    //If the image is continuous meaning it is being read like a flattend array
    if (frame1.isContinuous())
    {
        //Adjusts the rows and columns accordingly
        nCols *= nRows;
        nRows = 1;
    }

    //Defines the varaibles used as iterators
    int i, j;
    uchar* p1;
    uchar* p2;
    
    //Defines an x and y coordinate counter and defaults them to 0
    int x = 0, y = 0;

    //Loops through every row of the image
    for (i = 0; i < nRows; ++i)
    {
        //Creates a linked list for this row of pixels
        p1 = frame1.ptr<uchar>(i);
        p2 = frame2.ptr<uchar>(i);

        //Loops through every column of the image
        //Starting at 2 and incrmenting by 2 so we only consider red pixels
        for (j = 2; j < nCols; j += 3)
        {

            //Detects color in the frame
            if (type == 0) detectColor(p2, j, x, y, color, threshold);
            else detectMotion(p1, p2, j, x, y, threshold);

            //Incrments the x counter
            x++;
            //If the x is more than the frame width
            if (x >= frame1.cols)
            {
                //Increments the row counter and resets the x counter
                y++;
                x = 0;
            }

        }
    }
    //Writes the blobs to a text file
    for (Detect::blobIterator = Detect::blobs.begin();
         Detect::blobIterator != Detect::blobs.end(); 
         ++Detect::blobIterator)
    {
        //Writes the blob coordinates to a text file
        std::ofstream blobFile;
        blobFile.open("./blobCoords.txt");
        blobFile << blobIterator->getMinX() << ',' << 
                    blobIterator->getMinY() << ',' << 
                    blobIterator->getMaxX() << ',' << 
                    blobIterator->getMaxY() << std::endl;
        blobFile.close();

        //Diagnostic for displaying the blob bounding boxes
        /*
        cv::Rect rect(cv::Point(blobIterator->getMinX(), 
                                blobIterator->getMinY()), 
                      cv::Point(blobIterator->getMaxX(), 
                                blobIterator->getMaxY()));
        cv::rectangle(frame2, rect, cv::Scalar(0, 0, 255), -1);
        */
    }
    
}

void Detect::detectColor(uchar* p, int j, int x, int y, int color,
                         int threshold)
{
    //Calculates the distance of the red pixel
    int rDis = pow(p[j] - color, 2);

    //If our distance from black is less than our threshold we
    //cover the pixel with a red mesh
    if (rDis <= threshold)
    {
        //Sets the pixel to be red tinted
        p[j] = 255;

        //Initializes a flag for if the new pixel has been assigned to
        //a blob
        bool found = false;

        //Loops through the list of blobs
        for (Detect::blobIterator = Detect::blobs.begin(); 
            Detect::blobIterator != Detect::blobs.end();
            ++Detect::blobIterator)
        {
            //If the pixel distance is less than our threshold
            if (Detect::blobIterator->distanceTo(x, y) < Detect::blobThreshold)
            {
                //Adds the pixel to the blob and exits out of the loop
                Detect::blobIterator->addPx(x, y);
                found = true;
                break;
            }
        }

        //If the pixel was not added to a blob yet, adds it to one
        if (!found) Detect::blobs.push_back(Blob(x, y));
    }
}

void Detect::detectMotion(uchar* p1, uchar* p2, int j, int x, int y, int threshold)
{
    //Calculates the distance of the red pixel
    int rDis = pow(p1[j] - p2[j], 2);

    //If our distance from black is less than our threshold we
    //cover the pixel with a red mesh
    if (rDis >= threshold)
    {
        //Sets the pixel to be red tinted
        p2[j] = 255;

        //Initializes a flag for if the new pixel has been assigned to
        //a blob
        bool found = false;

        //Loops through the list of blobs
        for (Detect::blobIterator = Detect::blobs.begin();
            Detect::blobIterator != Detect::blobs.end();
            ++Detect::blobIterator)
        {
            //If the pixel distance is less than our threshold
            if (Detect::blobIterator->distanceTo(x, y) < Detect::blobThreshold)
            {
                //Adds the pixel to the blob and exits out of the loop
                Detect::blobIterator->addPx(x, y);
                found = true;
                break;
            }
        }

        //If the pixel was not added to a blob yet, adds it to one
        if (!found) Detect::blobs.push_back(Blob(x, y));
    }
}