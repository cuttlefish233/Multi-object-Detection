//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Eigen/Dense"

using namespace std;
using namespace cv;
using namespace Eigen;

Matrix2d getMice(Mat src, Mat flow);
void drawMice(Mat &image, Mat &src, Scalar boxColor); 

/*** Tools ****/
double getDistance(Point2d pointO, Point2d pointA);
Vector2d pointToVector(Point2d point);
//Put Text
void PutText(Mat &src, string text);
#pragma once
