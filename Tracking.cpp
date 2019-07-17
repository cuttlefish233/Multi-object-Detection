//C++ libraries
#include <vector>
#include <iostream>

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "utility.h"
#include "Eigen/Dense"
#include "KFilter.h"

using namespace std;
using namespace cv;
using namespace Eigen;

//Data Association Method. Since we know there are no more than 2 objects, we simplified the implementation here.
void HungarianMethod(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2, Mat &src, KFilter &kF1, KFilter &kF2)
{
	//One possible combination
	double d11 = getDistance(observation1, prediction1);
	double d22 = getDistance(observation2, prediction2);
	//Another possible combination
	double d12 = getDistance(observation1, prediction2);
	double d21 = getDistance(observation2, prediction1);

	cout << "d11: " << d11 << endl;
	cout << "d22: " << d22 << endl;
	cout << " ********************" << endl;
	cout << "d12: " << d12 << endl;
	cout << "d21: " << d21 << endl;
	cout << " ********************" << endl;

	int sum1 = getDistance(observation1, prediction1) + getDistance(observation2, prediction2);
	int sum2 = getDistance(observation1, prediction2) + getDistance(observation2, prediction1);

	Mat mouse1 = imread("mouse1.bmp", IMREAD_COLOR);
	Mat mouse2 = imread("mouse2.bmp", IMREAD_COLOR);

	//Hungarian Method, shortest overall distance
	if (sum1 <= sum2)
	{   //kF1 to mouse1, kF2 to mouse2
		drawMice(mouse1, src, kF1.getBoxColor());
		drawMice(mouse2, src, kF2.getBoxColor());
		kF1.update(pointToVector(observation1));
		kF2.update(pointToVector(observation2));
	}
	else //kF1 to mouse2, kF2 to mouse1
	{
		drawMice(mouse1, src, kF2.getBoxColor());
		drawMice(mouse2, src, kF1.getBoxColor());
		kF1.update(pointToVector(observation2));
		kF2.update(pointToVector(observation1));
	}
}