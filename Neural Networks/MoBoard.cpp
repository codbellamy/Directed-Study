/*********************************** HEADING **********************************
*																			  *
*								    Moboard.cpp								  *
*                            C. Bellamy  J. Steiner                           *
*																			  *
******************************************************************************/

//Includes the MoBoard header file
#include "MoBoard.h"

/****************************** TAKE FIX FUNCTION *****************************
* Name:  takeFix
* Param: bearing - the bearing of the contact to own ship
*        range   - the range of the contact to own ship
* Notes: converts from bearing and range (polar coordinates) to x and y
*        (cartesian coordinates)
******************************************************************************/
void Contact::takeFix(float bearing, float range, float min)
{
	//Calculates bearing dift based on the previous bearing and current bearing
	if (bearing == prevBearing)
		bearingDrift = "CBDR";
	else if (bearing >= prevBearing)
		bearingDrift = "RIGHT";
	else
		bearingDrift = "LEFT";

	//Finds the time between fixes
	fixInterval = min - fixInterval;

	//Defeaults booleans that x and y values are both positive distances.
	//This would mean that they are in the first quadrant (0-90 degrees T)
	bool xPos = true;
	bool yPos = true;

	//If we are in the second quadrant [270T, 360T)
	if (bearing >= 270)
	{
		//Decrements the bearing by 270
		bearing -= 270;

		//Sets the x value to be neagtive, since we are in the second quadrant
		xPos = false;
	}

	//If we are in the third quadrant [180T, 270T)
	else if (bearing >= 180)
	{
		//Decrements the bearing by 180
		bearing -= 180;

		//Sets the x and y values to be negative, since we are in the third
		//quadrant
		xPos = false;
		yPos = false;
	}

	//If we are in the fourth quadrant [90T, 180T)
	else if (bearing >= 90)
	{
		//Decrements the bearing by 90
		bearing -= 90;

		//Sets the y value to be negative, since we are in the fourth quadrant
		yPos = false;
	}

	//Converts the bearing from degrees to radians
	float bearingRad = bearing * DEGREE_TO_RAD;

	//Caculates the cartesian x and y values of the fix
	float x = cos(bearingRad) * range;
	float y = sin(bearingRad) * range;

	//Swaps the sign of the x and y values if applicable
	if (!xPos) x *= -1;
	if (!yPos) y *= -1;

	//Loops through the array looking for empty fixes space
	for (int i = MAX_FIXES - 1; i >= 0; i--)
	{
		//If the fix still has a default value
		if (fixes[i][0] < -pow(2, 10))
		{
			//Adds the x and y value to the fix
			fixes[i][0] = x;
			fixes[i][1] = y;

			//Break out of the loop since we found space
			break;
		}

		//If there was not space found in the loop and we are on our last loop
		//through to find space
		if (i == 0)
		{
			//Pushes all elements in the list down a peg
			for (int i = 0; i < MAX_FIXES - 1; i++)
			{
				fixes[i][0] = fixes[i + 1][0];
				fixes[i][1] = fixes[i + 1][1];
			}

			//Stores the new fix
			fixes[MAX_FIXES - 1][0] = x;
			fixes[MAX_FIXES - 1][1] = y;
		}
	}
}

/************************** CALC RANGE INFO FUNCTION **************************
* Name:  calcRange
* Notes: calculates relative motion info
******************************************************************************/
void Contact::calcRange(float heading)
{
	//Unpacks the last fix
	float x2 = fixes[0][0];
	float y2 = fixes[0][1];

	//Calculates the change in x and y in NM
	float deltaX = x2 - fixes[MAX_FIXES - 1][0];
	float deltaY = y2 - fixes[MAX_FIXES - 1][1];

	//Finds the distance between the first and last fixes plotted
	float distance = sqrt(pow(deltaX, 2) + pow(deltaY, 2));

	//Calculates the speed of relative motion based on distance and time
	srm = distance / ((fixInterval*2) / 60);

	//Finds the direction of relative motion, given we are in the first
	//quadrant
	drm = atan(abs(deltaY) / abs(deltaX)) * RAD_TO_DEGREE;

	//Defaults the offset in degrees to be 0
	int offset = 0;

	//If we are in the fourth quadrant, the offset should be 90 degrees
	if (deltaX >= 0 && deltaY <= 0) offset = 90;
	//If we are in the thrid quadrant, the offset should be 180 degrees
	else if (deltaX <= 0 && deltaY <= 0) offset = 180;
	//If we are in the second quadrant, the offset should be 270 degrees
	else if (deltaX <= 0 && deltaY >= 0) offset = 270;

	//Increments the direction of relative motion by the offset depending on
	//which quadrant we are plotting in
	drm += offset;

	/* Sets a default very large 'minimum distance' this is so most likely, any
	*  distance computed will be smaller (even when we compute distance 
	*  squared) so we can initialize the variable up here and modify it below */
	float minDistance = pow(2, 10);

	//Used to store the x and y value along the LRM that is at CPA
	float xCPA;
	float yCPA;

	//Looks at every 10th of a NM on either side of own ship from -2NM to 2NM
	for (float i = -2; i < 2; i += 0.1)
	{
		//Calculates the distance on the LRM from our lat fix to the point we
		//are checking
		float distToAx = abs(x2 - i);

		//Initializes the points of Ax and Ay (checking point A)
		float Ax;
		float Ay;

		//If our last fix was a positive x
		if (x2 >= 0)
		{
			//Must subtract distance to get to our point
			Ax = x2 - distToAx;

			//Calculates the y component of point A
			Ay = ((-distToAx * deltaY) / deltaX) + y2;
		}
			
		//If our last fix was a negative x
		else
		{
			//Must add distance to get to our point
			Ax = x2 + distToAx;

			//Calculates the y component of point A
			Ay = ((distToAx * deltaY) / deltaX) + y2;
		}
			
		//Finds the squared distance from point A to own ship
		float dis = pow(Ax, 2) + pow(Ay, 2);

		//If this squared distance is the smallest distance so far
		if (dis < minDistance)
		{
			//Updates the smallest distance so far
			minDistance = dis;

			//Updates the x and y components for cpa
			xCPA = Ax;
			yCPA = Ay;
		}
		
	}

	//Calculates the closest point of approach
	rCPA = sqrt(minDistance);

	//Calculates the bearing to CPA without being scaled into a 3D space
	bCPA = asin(xCPA / rCPA) * RAD_TO_DEGREE;

	//If we are in the quadrants where sine is a negative function
	if (y2 <= 0)
	{
		//Inverts the sign of bearing
		bCPA *= -1;

		//Adds ships head
		bCPA += heading;
	}
	//If we are in the quadrants where sine is a positive function
	else
		//Subtracts ships head
		bCPA -= heading;

	//While bearing is negative
	while (bCPA < 0)
		//Adds 360 to get it in our scope
		bCPA += 360;

	//While bearing is over 360
	while (bCPA >= 360)
		//Subtracts 360 to get it in our scope
		bCPA -= 360;

	//Calculates the time until CPA
	tCPA = (sqrt(pow(xCPA - x2, 2) + pow(yCPA - y2, 2)) * 60) / srm;

	
}

/****************************** PLOTTING FUNCTION *****************************
* Name:   plot
* Param:  frame - the frame to plot the fixes to
* Return: frame - the frame after it has been plotted to
* Notes:  plots the fixes onto the frame
******************************************************************************/
cv::Mat Contact::plot(cv::Mat frame)
{
	//Calculates and stores the center of the maneuvering board
	int centerX = (frame.cols / 2);
	int centerY = (frame.rows / 2);

	//Loops through each fix in the cartesian fixes stored
	for (int i = MAX_FIXES - 1; i >= 0; i--)
	{
		//Converts from NM to pixels
		int displayX = ((fixes[i][0]  * frame.cols) / 20) + centerX;
		int displayY = ((-fixes[i][1] * frame.rows) / 20) + centerY;

		//Plots the fix to the frame
		cv::circle(frame, cv::Point(displayX, displayY), 5, cv::Scalar(0, 0, 255), -1);
	}

	//Calculates the pixel values of the last frame and the first frame
	int x1 = (( fixes[MAX_FIXES - 1][0] * frame.cols) / 20)      + centerX;
	int y1 = ((-fixes[MAX_FIXES - 1][1] * frame.rows) / 20)      + centerY;
	int x2 = (( fixes[0][0]             * frame.cols) / 20)      + centerX;
	int y2 = ((-fixes[0][1]             * frame.rows) / 20)      + centerY;

	//Defaults the change in y to be 0
	int rise = 0;

	/* We need to account for how x and y positions are plotted in a computer
	*  image vs a coordinate plane below. we plot in a computer image as if we 
	*  were in an inverted first quadrant. the below change to slope appears to
	*  work well in compensating for this. */

	//Change in y should be inverted
	rise = -1 * (y2 - y1);

	//Calculates the change in x
	int run = x2 - x1;

	/* To draw a relative motion vector, we need to calculate a faroff 
	*  'future point' on the same line or relative motion to calculate
	*  this 'future point' we increment the last fix by 50% of the screen size
	*  in pixels */

	//Defaults the future x and y points to be -1, a flag to see if they are
	//updated throughout the function
	int nextX = -1;
	int nextY = -1;

	//If there is a positive change in x
	if (run > 0)
		//Increments x2 by another 50% of screen space
		nextX = x2 + centerX;

	//If there is a negative change in x
	else if (run < 0)
		//Decrements x2 by another 50% of screen space
		nextX = x2 - centerX;

	//If there is no change in x
	else
	{
		//If there is a positive change in y
		if (rise > 0)
			//Increments y2 by 50% of the screen space
			nextY = y2 + centerY;
		//If there is a nonpositive change in y
		else
			//Decrements y2 by 50% of the screen space
			nextY = y2 - centerY;

		//Sets the next X value to the current x value and the rise to be not 
		//equal to 0 so we don't get a divide by 0 error
		nextX = x2;
		run = 0.01;
	}

	//If the future y has not been set yet
	if (nextY == -1)
		//Sets the future y based off the future x
		nextY = y2 - ((nextX-x2) * rise) / run;

	//Draws an arrowed line indicating the line of relative mtoin
	cv::arrowedLine(frame, cv::Point(x1, y1), cv::Point(nextX, nextY), 
					cv::Scalar(255, 0, 0), 3);

	//Returns the frame that was plotted on
	return frame;
}

/****************************** GET FIX FUNCTION ******************************
* Name:   getFix
* Param:  element - the index of the fix to return
*         key     - whether to return the x or the y component of the fix
* Return: fixes[element][key]
* Notes:  returns the x or y component of a fix
******************************************************************************/
float Contact::getFix(int element, int key)
{
	//Returns the x or y compoenet of a fix
	return fixes[element][key];
}

/****************************** GET SRM FUNCTION ******************************
* Name:   getSRM
* Return: srm
* Notes:  returns the speed of relative motion of the contact
******************************************************************************/
float Contact::getSRM()
{
	//Returns the speed of relative motion of the contact
	return srm;
}

/****************************** GET DRM FUNCTION ******************************
* Name:   getDRM
* Return: drm
* Notes:  returns the direction of relative motion of the contact
******************************************************************************/
float Contact::getDRM()
{
	//Returns the direction of relative motion of the contact
	return drm;
}

/****************************** GET RCPA FUNCTION *****************************
* Name:   getRCPA
* Return: rCPA
* Notes:  returns the range at the contact's closest point of approach
******************************************************************************/
float Contact::getRCPA()
{
	//Returns the range at the contact's closest point of approach
	return rCPA;
}

/****************************** GET BCPA FUNCTION *****************************
* Name:   getBCPA
* Return: bCPA
* Notes:  returns the bearing at the contact's closest point of approach
******************************************************************************/
float Contact::getBCPA()
{
	//Returns the range at the contact's closest point of approach
	return bCPA;
}

/****************************** GET TCPA FUNCTION *****************************
* Name:   getTCPA
* Return: tCPA
* Notes:  returns the time until the contact's closest point of approach from 
*         the last fix in minutes
******************************************************************************/
float Contact::getTCPA()
{
	//Returns the time unitl the contact's closest point of approach from the 
	//last fix in minutes
	return tCPA;
}

/************************** GET BEARING DRIFT FUNCTION ************************
* Name:   getBearingDrift
* Return: bearingDrift
* Notes:  returns the bearing drift of the contact
******************************************************************************/
std::string Contact::getBearingDrift()
{
	//Returns the bearing drift of the contact
	return bearingDrift;
}