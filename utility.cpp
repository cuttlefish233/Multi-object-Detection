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

Matrix2d getMice(Mat src, Mat flow) {
	//Result
	Matrix2d centroid_s; 
	int erosion_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(erosion_size, erosion_size));
	dilate(flow, flow, element);
	bitwise_not(flow, flow);
	Mat src_gray, thres_output, result1;
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	Rect area = Rect(0.187*src.cols, 0.2*src.rows, 0.7*src.cols, 0.739*src.rows);
	mask(area).setTo(255);
	bitwise_not(mask, mask);
	//imshow("mask", src);
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	src_gray.setTo(255, flow);
	src_gray.setTo(255, mask);
	threshold(src_gray, thres_output, 70, 255, 0);

	bitwise_not(thres_output, thres_output);
	cv::erode(thres_output, thres_output, element);
	//cv::erode(thres_output, thres_output, element);
	cv::dilate(thres_output, thres_output, element);

	imshow("Thres", thres_output);

	cv::Mat  labels, stats, centroids;
	//Mat image = Mat::zeros(src.rows, src.cols, CV_8UC1);
	int nccomps = cv::connectedComponentsWithStats(
		thres_output, labels,
		stats, centroids
	);

	if (nccomps >= 2) {
		//find largest 2 components and draw rectangles
		int max1 = 1;
		int max2 = 2;

		if (stats.at<int>(max2, cv::CC_STAT_AREA) > stats.at<int>(max1, cv::CC_STAT_AREA)) {
			max1 = 2;
			max2 = 1;
		}
		for (int i = 3; i < nccomps; i++) {
			if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(max1, cv::CC_STAT_AREA)) {
				max2 = max1;
				max1 = i;
			}
			else if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(max2, cv::CC_STAT_AREA)) {
				max2 = i;
			}
		}

		Mat src2, src3;
		src.copyTo(src2);
		src.copyTo(src3);

		double x = centroids.at<double>(max1, 0);
		double y = centroids.at<double>(max1, 1);
		//cout << "x: " << x << endl;
		//cout << "y: " << y << endl;
		floodFill(src2, Point(x, y), Scalar(0, 0, 0), 0, Scalar(2, 1, 1), Scalar(5, 5, 5), 8);
		//floodFill(canny, Point(x,y), 155, 0, 2, 2, 8);
		double x2 = centroids.at<double>(max2, 0);
		double y2 = centroids.at<double>(max2, 1);
		//cout << "x2: " << x2 << endl;
		//cout << "y2: " << y2 << endl;
		floodFill(src3, Point(x2, y2), Scalar(0, 0, 0), 0, Scalar(2, 1, 1), Scalar(5, 5, 5), 8);

		for (int i = 0; i < labels.cols; i++) {
			for (int j = 0; j < labels.rows; j++) {
				int label = labels.at<int>(j, i);
				Vec3b intensity = src.at<Vec3b>(j, i);
				if (label == max1) {
					src2.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
				}
				if (label == max2) {
					src3.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
				}

			}
		}

		imwrite("mouse1.bmp", src2);
		imwrite("mouse2.bmp", src3);

		centroid_s << centroids.at<double>(max1, 0), centroids.at<double>(max1, 1),
			centroids.at<double>(max2, 0), centroids.at<double>(max2, 1);
	}

	return centroid_s;
}

void drawMice(Mat &image, Mat &src, Scalar boxColor) {
	cv::Mat  labels, stats, centroids;
	cvtColor(image, image, CV_BGR2GRAY);
	threshold(image, image, 0, 255, 0);
	bitwise_not(image, image);
	int nccomps = cv::connectedComponentsWithStats(
		image, labels,
		stats, centroids
	);
	//: Find largest component
	int maxsize = 0;
	int maxind = 0;
	for (int i = 1; i < nccomps; i++)
	{
		// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
		double area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area > maxsize) {
			maxsize = area;
			maxind = i;
		}
	}
	double left, top, height, width;
	left = stats.at<int>(maxind, CC_STAT_LEFT);
	top = stats.at<int>(maxind, CC_STAT_TOP);
	width = stats.at<int>(maxind, CC_STAT_WIDTH);
	height = stats.at<int>(maxind, CC_STAT_HEIGHT);
	Rect rect1(left, top, width, height);
	rectangle(src, rect1, boxColor, 5, 8, 0);
}

/***** Distance between two points *****/
double getDistance(Point2d pointO, Point2d pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	return distance;
}

//Data Association Method. Since we know there are no more than 2 objects, we simplified the implementation here.
/*
Vector2d Hungarian(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2) {
	int sum1 = getDistance(observation1, prediction1) + getDistance(observation2, prediction2);
	int sum2 = getDistance(observation1, prediction2) + getDistance(observation2, prediction1);

	Vector2d result;
	if (sum1 <= sum2)
		result << 1, 1;
	else
		result << 1, 2;

	return result;
}
*/

Vector2d pointToVector(Point2d point) {
	Vector2d vector;
	vector << point.x, point.y;
	return vector;
}

void PutText(Mat &src, string text) {
	Point origin(0, 100);
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 0.5;
	int thickness = 1;
	putText(src, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
}