#include <iostream> 
#include <queue>

#include "Eigen/Dense"
#include "utility.h"

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

class KFilter
{
private:
	//Color of the boundingBox
	Scalar boxColor;

	//2 measures: x and y
	const int measureNum = 2;
	//4 states: x_t, y_t, vx_t, vy_t, constant velocity value
	const int stateNum = 4;
	//The delta of time, we assume the video is 30fps
	//Maybe its not right to use fps as the deltaT
	//Maybe I should just assume a value?
	double deltaT = 1 / 30;
	//Acceleration
	const double u = 4e-3;

	// initial value of X
	Vector4d x_init;

	//The measurement of X
	//Vector2d z;

	//Identity Matrix I
	const Matrix4d identityMatrix = MatrixXd::Identity(stateNum, stateNum);

	// Kalman filter variables
	//Now we assumen that A, B, H are all identity
	//A: Transition Matrix
	Matrix4d A;
	Vector4d B;
	MatrixXd H;

	//Q and R are White Gaussian Noise
	//We all use 0.1 as the initial value, may adjust later
	//measurement noise covariance matrix, ����R=E(V2j), �ǲ���������Э����(��), ��ϵͳ��ͼ�е� Vj ��Э����, Ϊ�˼򻯣�Ҳ����һ����������; 
	//const double init_measure_uncertainty = 1;
	Matrix2d R;

	//process noise covariance matrix, Q=E(W2j) ��ϵͳ������Э�����󣬼�ϵͳ��ͼ�е�Wj��Э������, Q Ӧ���ǲ��ϱ仯�ģ�Ϊ�˼򻯣�����һ����������
	Matrix4d Q;

	//state variables covariance matrix
	//const double init_state_variables_uncertainty = 10;
	Matrix4d P_init;

	/** Variables used inside the process **/
	//xk_1: value of state vector x at k-1th frame
	Matrix4d Pk_1;
	Vector4d xk_1;

	/**
	x_k: value of state vector X at k-th frame
	P_k: (a posteriori)Error covariance matrix k
	K_k: value of Kalman gain at k-th frame
	*/
	Vector4d x_k;
	Matrix4d P_k;
	MatrixXd K_k;

	queue<Point2d> lastPositions;

public:
	KFilter(Scalar boxColor);
	~KFilter();
	void initialize();
	Point2d predict();
	/**
	 z: Measurement(observation), the coordinate of the object
	*/
	void update(Vector2d z);
	//set x
	void setInitialState(double x, double y);
	queue<Point2d> getLastPositions();
	Scalar getBoxColor();
};
#pragma once
