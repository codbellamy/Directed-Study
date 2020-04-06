/********************************** HEADING ***********************************
*																			  *
*									Detect.h 								  *
*							 J. Steiner   C. Bellamy                          *
*																			  *
******************************************************************************/

/***************************** LOAD DEPENDENCIES *****************************/

//Ensures that the header file and its dependencies will only be included once
//even if it shows up multiple times throughout the program
#pragma once

//Includes the ability to load and work with images
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//The ability to print diagnostics to the console
#include <iostream>

//The ability to work with arrays of variable and non constant size
#include <list>

//The ability to write to text files
#include <fstream>

/***************************** DEFINES STRUCTURES ****************************/

//Defines a new data type, a single 3D point within an image handled as
//an array of 8 bit unsigned integers we will call a pixel
typedef cv::Point3_<uint8_t> Pixel;

/********************************* BLOB CLASS ********************************/

class Blob
{

	//Public attributes to the Blob class
	public:

		//Blob class constructor
		Blob(int, int);

		//The distance to the point inputted to the blob edges
		int distanceTo(int, int);

		//Adds a pixel to the blob
		void addPx(int, int);

		//Gets the extreme coordinates of the blob
		int getMaxX();
		int getMaxY();
		int getMinX();
		int getMinY();

	//Private attributes to the Blob class
	private:

		//The extreme coordinates of the blob
		int maxX, minX, maxY, minY;
};

/******************************** DETECT CLASS *******************************/

class Detect
{

	//Public attributes to the detect class
	public:

		//The class constructor
		Detect(int);

		//Conducts a type of detection to the image
		void searchFrame(cv::Mat&, cv::Mat&, int, int, int);
		

	//Private attributes to the detect class
	private:

		void detectColor(uchar*, int, int, int, int, int);

		void detectMotion(uchar*, uchar*, int, int, int, int);

		//Blob distance threshold
		int blobThreshold;

		//Defines an object list and an iterator through that object list
		std::list <Blob> blobs;
		std::list<Blob>::iterator blobIterator;
};