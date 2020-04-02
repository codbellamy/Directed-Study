/*********************************** HEADING **********************************
*																			  *
*								    Moboard.h								  *
*                            J. Steiner  C. Bellamy                           *
*																			  *
******************************************************************************/

//Ensures that this file will only be included once
#pragma once

//Includes opencv resources
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//Includes other resources
#include <iostream>
#include <string.h>
#include <math.h>


//Defines a constant converting degrees into radians
const float DEGREE_TO_RAD = 3.14159 / 180;

//Defines a constant converting radians into degrees
const float RAD_TO_DEGREE = 180 / 3.1459;

//Defines how many fixes we will store at a maximum
const int MAX_FIXES = 3;

/****************************** CONTACT CLASS *******************************/
//Stores values for each contact
class Contact
{
	//All public variables, methods, and functions
	public:

		//Defines a constant that converts bearing and range 
		//(polar coordinates) into x and y (cartesian coordinates)
		void takeFix(float, float, float);

		//Calculates range info
		void calcRange(float);

		//Plots the contact's fixes and a line of relative motion to connect them
		cv::Mat plot(cv::Mat);

		//Gets the contact's fixes
		float getFix(int, int);

		//Gets the contact's srm
		float getSRM();

		//Gets the contact's drm
		float getDRM();

		//Gets the contact's range of CPA
		float getRCPA();

		//Gets the contact's bearing of CPA
		float getBCPA();

		//Gets the contact's time until CPA (from last fix)
		float getTCPA();

		//Gets the contact's bearing drift
		std::string getBearingDrift();

	//All private variables, methods, and functions
	private:

		//An array of cartesian fixes
		float fixes[MAX_FIXES][2];

		//Stores the speed of relative motion
		float srm;

		//Stores the direction of relative motion
		float drm;

		//Stores the range of CPA
		float rCPA;

		//Stores the bearing of CPA
		float bCPA;

		//Stores the time until CPA in minutes
		float tCPA;

		//Stores the bearing drift of the contact
		std::string bearingDrift;

		//Stores the previous fixes bearing to calculate bearing drift
		float prevBearing;

		//The time between fixes
		float fixInterval = 0;
};