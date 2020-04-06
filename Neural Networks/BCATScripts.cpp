/********************************** HEADING ***********************************
*                                                                             *
*                                BCATScripts.cpp                              *
*							 C. Bellamy   J. Steiner                          *
*                                                                             *
******************************************************************************/

/***************************** LOAD DEPENDENCIES *****************************/

//Includes the ability to run detection algorithms
#include "Detect.h"

//Includes the ability to load and process video
#include <opencv2/videoio.hpp>

/******************************* MAIN FUNCTION *******************************/
int main(int argc, char* argv[])
{
    //Initializes a matrix object to load each frame
    cv::Mat frame1, frame2;
    //Diagnositic for getting sample frames
    /*
    cv::VideoCapture cap("./videos/sail_boat_small.mp4");

    for (int i = 0; i < 10; i++)
    {
        cap >> frame1;
        cap >> frame2;
    }
    cap.release();
    cv::imwrite("./frame1.jpg", frame1);
    cv::imwrite("./frame2.jpg", frame2);
    */

    //Loads in the frame from a local file
    frame1 = cv::imread("./frame1.jpg");
    frame2 = cv::imread("./frame2.jpg");

    //If one of the frames did not load correctly, exit with an error code
    if (frame1.empty() || frame2.empty()) return -1;
        
    //If not all arguments are given, use defaults
    if (argc < 4)
    {
        //Creates a detect object
        Detect d(122500);

        //Conducts detection
        d.searchFrame(frame1, frame2, 0, 2000, 1);

        //Writes our detected frame to a file
        cv::imwrite("./processedFrame.jpg", frame2);
    }

    //If all arguments are given, use them
    else
    {
        //Loads the argumenst
        std::string blobThreshold = argv[1];
        std::string detectThreshold = argv[2];
        std::string detectType = argv[3];

        //Creates a detect object
        Detect d(stoi(blobThreshold));

        //Conducts detection
        d.searchFrame(frame1, frame2, 0, stoi(detectThreshold), 
                      stoi(detectType));

        //Writes our detected frame to a file
        cv::imwrite("./processedFrame.jpg", frame2);
    }
    

    //The program exited successfully
    return 0;
}