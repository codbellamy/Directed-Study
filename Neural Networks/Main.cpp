/*********************************** HEADING **********************************
*																			  *
*									Main.cpp								  *
*                            C. Bellamy  J. Steiner                           *
*																			  *
******************************************************************************/

/****************************** INCLUDE HEADERS ******************************/
#include <string>
#include <fstream>
#include "MoBoard.h"

/******************************* MAIN FUNCTION *******************************/
int main()
{
	//Initializes a variable to hold the fixes read
	std::string fix;
	//Initializes a variable to hold the course and speed of ownship
	std::string ownship;

	//Reads in the fixes from a file
	std::ifstream fixes ("./fixes.txt");

	//Defines a class to store contact fixes
	Contact m;

	//Loads in a background image of a maneuvering board
	cv::Mat moboardBG = cv::imread("./images/moboard.jpg");

	//If the image loading from file failed, send an error message
	if (!moboardBG.data) std::cout << "MoBoard BG failed to load" << std::endl;

	//If the file was loaded in successfully
	if (fixes.is_open())
	{
		//While the line can be retrieved, runs the lower code block
		//loads in the next line of the text file and stores it in the string
		//called fix
		while (getline(fixes, fix))
		{
			//Finds the index of the end of bearing in the line
			size_t endofBearing = fix.find(',');
			//Finds the index of the end of the range in the line
			size_t endofRange   = fix.find(',', endofBearing+1)-endofBearing-1;
			
			//Loads in the bearing of the fix from the line
			float bearing = std::stof(fix.substr(0, endofBearing));
			//Loads in the range of the fix from the line
			float range = std::stof(fix.substr(endofBearing+1, endofRange));
			//Loads in the time of the fix from the line
			float time = std::stof(fix.substr(endofRange+1+endofBearing+1));

			//Converts bearing and range (polar coordinates) to x and y
			//(cartesian coordinates)
			m.takeFix(bearing, range, time);
		}
		
		//Calculates range info
		m.calcRange(90);

		//Outputs range info to the console
		std::cout << "SRM: " << m.getSRM() << " KTS" << std::endl;
		std::cout << "DRM: " << m.getDRM() << " T" << std::endl;
		std::cout << "RCPA: " << m.getRCPA() << " NM" << std::endl;
		std::cout << "BCPA: " << m.getBCPA() << " R" << std::endl;
		std::cout << "Time unitl CPA: " << m.getTCPA() << " mins" << std::endl;
		std::cout << "Bearing Drift: " << m.getBearingDrift() << std::endl;

		//Plots the fixes and range info on the maneuvering board
		cv::Mat moboard = m.plot(moboardBG);

		//Dispalys the moboard image and waits for input
		cv::imshow("MoBoard", moboard);
		cv::waitKey(0);

		//Deallocates memory to the fixes text file
		fixes.close();
	}

	//If the file did not open correctly, prints an error message
	else std::cout << "Unable to open file" << std::endl;

	//Returns that the program exited successfully
	return 0;
}