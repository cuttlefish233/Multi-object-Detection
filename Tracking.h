//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "KFilter.h"

using namespace std;
using namespace cv;

//Hungarian Method
//Vector2d Hungarian(Point2d observations1, Point2d observations2, Point2d prediction1, Point2d prediction2);
//Data Association Method(Hungarian + error position process)
void HungarianMethod(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2, Mat &src, KFilter &kF1, KFilter &kF2);
#pragma once
