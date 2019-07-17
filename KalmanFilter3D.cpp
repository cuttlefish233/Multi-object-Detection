//C++ libraries
#include <iostream> 
#include <cmath>

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//Custom Libraries
#include "Eigen/Dense"
#include "KalmanFilter3D.h"

using namespace Eigen;
using namespace std;
using namespace cv;

KalmanFilter3D::KalmanFilter3D(Scalar boxColor) {
	this->boxColor = boxColor;
	initialize();
}

KalmanFilter3D::~KalmanFilter3D() {

}

void KalmanFilter3D::initialize() {
	Pk_1 = MatrixXd(6, 6);
	xk_1 = MatrixXd(6, 6);
	x_k = MatrixXd(6, 6);
	P_k = MatrixXd(6, 6);
	K_k = MatrixXd(6, 3);

	A = MatrixXd(6, 6);
	A << 1, 0, 0, deltaT, 0, 0,
		0, 1, 0, 0, deltaT, 0,
		0, 0, 1, 0, 0, deltaT,
		0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1;
	
	B = MatrixXd(6, 6);
	B << pow(deltaT, 2) / 2, pow(deltaT, 2) / 2, pow(deltaT, 2) / 2, deltaT, deltaT, deltaT;

	H = MatrixXd(3, 6);
	H << 1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0;

	R << 0.1, 0, 0,
		0, 0.1, 0,
		0, 0, 0.1;

	Q = MatrixXd(6, 6);
	//Q = B * BT
	/*
	Q << pow(deltaT, 4) / 4, pow(deltaT, 4) / 4, pow(deltaT, 3) / 2, pow(deltaT, 3) / 2,
		pow(deltaT, 4) / 4, pow(deltaT, 4) / 4, pow(deltaT, 3) / 2, pow(deltaT, 3) / 2,
		pow(deltaT, 3) / 2, pow(deltaT, 3) / 2, pow(deltaT, 2), pow(deltaT, 2),
		pow(deltaT, 3) / 2, pow(deltaT, 3) / 2, pow(deltaT, 2), pow(deltaT, 2);
		*/

	Q << 0.1, 0, 0, 0, 0, 0,
		0, 0.1, 0, 0, 0, 0,
		0, 0, 0.1, 0, 0, 0,
		0, 0, 0, 0.1, 0, 0,
		0, 0, 0, 0, 0.1, 0,
		0, 0, 0, 0, 0, 0.1;

	P_init = MatrixXd(6, 6);
	P_init << 0.1, 0, 0, 0, 0, 0,
		0, 0.1, 0, 0, 0, 0,
		0, 0, 0.1, 0, 0, 0,
		0, 0, 0, 0.1, 0, 0,
		0, 0, 0, 0, 0.1, 0,
		0, 0, 0, 0, 0, 0.1;

	Pk_1 = P_init;
	//X_1 all zero for start, could change later
	xk_1 << 0, 0, 0, 0, 0, 0;
}

Point3d KalmanFilter3D::predict() {
	//predict (Time update)
	x_k = A * xk_1 + B * u; //(A * xk_1 + Buk)
	P_k = A * Pk_1 * A + Q;  //(A * Pk_1 * AT + Q)
	return Point3d(x_k(0), x_k(1), x_k(2));
}

void KalmanFilter3D::update(Vector3d z) {
	//update (Measurement Update)
	MatrixXd S_k = H * P_k * H.transpose() + R;
	K_k = P_k * H.transpose() * S_k.inverse();  //(P_k * HT) * [(H * P_k * HT + R)]-1
	x_k = x_k + K_k * (z - H * x_k);
	P_k = (identityMatrix - K_k * H) * P_k;   //(I - K_k * H) * P_k

	//x_k is what we want
	xk_1 = x_k;
	Pk_1 = P_k;
}

Scalar KalmanFilter3D::getBoxColor() {
	return boxColor;
}