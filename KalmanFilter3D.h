#include <iostream> 
#include "Eigen/Dense"

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

class KalmanFilter3D
{
private:
	//Color of the boundingBox
	Scalar boxColor;

	//3 measures: x, y, z
	const int measureNum = 3;
	//6 states: x_t, y_t, z_t, vx_t, vy_t, vz_t, constant velocity value model
	const int stateNum = 6;
	//The delta of time, we assume the video is 30fps
	//Maybe its not right to use fps as the deltaT
	//Maybe I should just assume a value?
	double deltaT = 1 / 30;
	//Acceleration
	//const double u = 4e-3;
	const double u = 0;

	// initial value of X
	Vector4d x_init;

	//The measurement of X
	//Vector2d z;

	//Identity Matrix I
	const MatrixXd identityMatrix = MatrixXd::Identity(stateNum, stateNum);

	// Kalman filter variables
	//Now we assumen that A, B, H are all identity
	//A: Transition Matrix
	MatrixXd A;
	VectorXd B;
	MatrixXd H;

	//Q and R are White Gaussian Noise
	//We all use 0.1 as the initial value, may adjust later
	//measurement noise covariance matrix, 这里R=E(V2j), 是测量噪声的协方差(阵), 即系统框图中的 Vj 的协方差, 为了简化，也当作一个常数矩阵; 
	//const double init_measure_uncertainty = 1;
	Matrix3d R;

	//process noise covariance matrix, Q=E(W2j) 是系统噪声的协方差阵，即系统框图中的Wj的协方差阵, Q 应该是不断变化的，为了简化，当作一个常数矩阵。
	MatrixXd Q;

	//state variables covariance matrix
	//const double init_state_variables_uncertainty = 10;
	MatrixXd P_init;

	/** Variables used inside the process **/
	//xk_1: value of state vector x at k-1th frame
	MatrixXd Pk_1;
	VectorXd xk_1;

	/**
	x_k: value of state vector X at k-th frame
	P_k: (a posteriori)Error covariance matrix k
	K_k: value of Kalman gain at k-th frame
	*/
	VectorXd x_k;
	MatrixXd P_k;
	MatrixXd K_k;

public:
	KalmanFilter3D(Scalar boxColor);
	~KalmanFilter3D();
	void initialize();
	Point3d predict();
	/**
	 z: Measurement(observation), the position of the object
	*/
	void update(Vector3d z);
	Scalar getBoxColor();
};

#pragma once
