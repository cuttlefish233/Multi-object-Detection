//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

//C++ standard libraries
#include <io.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <fstream>

//Eigen Library
#include "Eigen/Dense"

//Custom Library
#include "utility.h"
#include "KFilter.h"
#include "Tracking.h"

using namespace std;
using namespace cv;
using namespace Eigen;

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, Mat& zero, int step,
	double, const Scalar& color)
{
	int angle1 = -1;
	float threshld = 0.2;
	for (int y = 0; y < cflowmap.rows; y += 1)
		for (int x = 0; x < cflowmap.cols; x += 1)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			if (abs(fxy.x) > threshld || abs(fxy.y) > threshld) {
				float param = fxy.y / fxy.x;
				zero.at<uchar>(y, x) = 255;
			}
		}
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), Scalar(0, 255, 0));
		}
	//imshow("flow", cflowmap);
}

int main() {

	KFilter kF1(Scalar(255, 0, 0));
	kF1.setInitialState(495.225, 618.779);
	KFilter kF2(Scalar(0, 255, 0));
	kF2.setInitialState(333.998, 316.763);
	VideoCapture capture("./1.avi");
	//Mat frame;
	if (!capture.isOpened())
	{
		printf("can not open ...\n");
		return -1;
	}

	Mat flow, cflow, frame;
	UMat gray, prevgray, uflow;
	int frameCount = 0;
	ofstream f1("log.txt");

	while (1)
	{
		capture >> frame;
		if (frame.empty())
		{
			cout << "Frame is empty!" << endl;
			break;
		}
		resize(frame, frame, Size(640, 800));

		Matrix2d centroids;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		Mat zero = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		if (!prevgray.empty())
		{
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
			uflow.copyTo(flow);
			drawOptFlowMap(flow, cflow, zero, 5, 1.5, Scalar(255, 255, 255));
			centroids = getMice(frame, zero);
			//imshow("flowFilter", zero);
		}

		Mat frame1 = frame.clone();

		PutText(frame1, std::to_string(frameCount));


		//Position of mouse1
		Point2d observation1 = Point2d(centroids(0, 0), centroids(0, 1));
		//Position of mouse2
		Point2d observation2 = Point2d(centroids(1, 0), centroids(1, 1));

		if (frameCount == 0)
		{
			observation1 = Point2d(0, 0);
			observation2 = Point2d(0, 0);
		}

		Point2d prediction1 = kF1.predict();
		Point2d prediction2 = kF2.predict();

		f1 << setw(20) << "Frame£º" << frameCount << endl;
		f1 << setw(20) << "Observations1£º" << observation1 << endl;
		f1 << setw(20) << "Observations2£º" << observation2 << endl;
		f1 << setw(20) << "Prediction1£º" << prediction1 << endl;
		f1 << setw(20) << "Prediction2£º" << prediction2 << endl;
		f1 << endl;

		//Data Association
		HungarianMethod(observation1, observation2, prediction1, prediction2, frame1, kF1, kF2);

		namedWindow("Result", WINDOW_AUTOSIZE);
		imshow("Result", frame1);
		imwrite("./output/" + std::to_string(frameCount) + ".bmp", frame1);
		waitKey(10);  //Ã¿Ò»Ö¡ÑÓ³Ù10ºÁÃë
		std::swap(prevgray, gray);
		frameCount++;
	}
	f1.close();
	capture.release();
	return 0;
}