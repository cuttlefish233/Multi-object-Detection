#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#define PI 3.14159265
using namespace cv;
using namespace std;
void drawMice(Mat &image, Mat &src);
void getMice(Mat src, Mat thres);
void MiceDetect(Mat& src, Mat& dst);
bool IfMice(Vec3b intensity);

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
	imshow("flow", cflowmap);
}
int main(int argc, char** argv)
{
	VideoCapture cap;
	cap.open("1.avi");
	if (!cap.isOpened())
		return -1;

	Mat flow, cflow, frame;
	UMat gray, prevgray, uflow;
	namedWindow("flow", 1);

	while (cap.read(frame))
	{
		//cap >> frame;
		resize(frame, frame, Size(500, 300));
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//blur(gray, gray, Size(3, 3));
		//blur(gray, gray, Size(3, 3));
		Mat zero = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		if (!prevgray.empty())
		{
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
			uflow.copyTo(flow);
			drawOptFlowMap(flow, cflow, zero, 5, 1.5, Scalar(255, 255, 255));
			getMice(frame, zero);
			imshow("flowFilter", zero);
		}
		if (waitKey(30) >= 0)
			break;
		std::swap(prevgray, gray);
		//cap.read(frame);
	}
	return 0;
}
void getMice(Mat src, Mat flow) {
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


	//Mat thresout;
	//resize(thres_output,thresout,Size(500, 300));
	//bitwise_and(thres_output, mask, thres_output);
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
		floodFill(src2, Point(x, y), Scalar(0, 0, 0), 0, Scalar(2, 1, 1), Scalar(5, 5, 5), 8);
		//floodFill(canny, Point(x,y), 155, 0, 2, 2, 8);
		double x2 = centroids.at<double>(max2, 0);
		double y2 = centroids.at<double>(max2, 1);
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

		drawMice(src2, src);
		drawMice(src3, src);
	}

	//bool f1 = true, f2 = true;

	//resize(src,src,Size(500, 300));
	imshow("result2", src);
}
void drawMice(Mat &image, Mat &src) {
	cv::Mat  labels, stats, centroids;
	cvtColor(image, image, CV_BGR2GRAY);
	threshold(image, image, 0, 255, 0);
	bitwise_not(image, image);
	//resize(image, srcout, Size(500, 300));
	//imshow("iii", srcout);
	int nccomps = cv::connectedComponentsWithStats(
		image, labels,
		stats, centroids
	);
	//: Find largest contour
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
	rectangle(src, rect1, Scalar(0, 255, 0), 5, 8, 0);
}
//Function that returns the maximum of 3 integers

bool IfMice(Vec3b intensity) {
	int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
	if (abs(R - 30) < 5 && abs(G - 20) < 5 && abs(B - 10) < 5) {
		return true;
	}
	else if (abs(R - 20) < 10 && abs(G - 10) < 5 && abs(B - 10) < 5) {
		return true;
	}
	return false;
}
void MiceDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			////part between shadow and bright
			if (abs(R - 30) < 5 && abs(G - 20) < 5 && abs(B - 10) < 5) {
				dst.at<uchar>(i, j) = 255;
			}
			if (abs(R - 20) < 10 && abs(G - 10) < 5 && abs(B - 10) < 5) {
				dst.at<uchar>(i, j) = 255;
			}
			////bright part of hands
			//if ((R > 170 && G > 100 && B > 70) && (myMax(R, G, B) - myMin(R, G, B) > 60) && (abs(R - G) > 50) && abs(G - B) > 20 && (R > G) && (G > B)) {
			//	dst.at<uchar>(i, j) = 255;
			//}
			//shadow on hands
			/*if ((R < 35 && G < 25 && B < 20) && (myMax(R, G, B) - myMin(R, G, B) < 20) && (abs(R - G) <10) && abs(G - B) <10) {
				dst.at<uchar>(i, j) = 255;
			}*/
		}
	}
}
