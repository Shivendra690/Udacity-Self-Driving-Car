#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#define EPS 0.0001 // a very small number

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // State dimension
  n_x_ = x_.size();

  // Augmented state dimension
  n_aug_  = n_x_ + 2;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Weights of sigma points
  weights_ = VectorXd(n_sig_);

  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_laser_ = 0.0;
  
  // Radar measurement noise covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << pow(std_radr_, 2), 0, 0,
              0, pow(std_radphi_, 2), 0,
              0, 0, pow(std_radrd_, 2);
  
  // Lidar measurement noise covariance matrix
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << pow(std_laspx_, 2), 0,
              0, pow(std_laspy_, 2);

  is_initialized_ = false;

}

UKF::~UKF() {}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {

    // initialize weights
    double lambda_n_aug = lambda_+n_aug_;
    weights_(0) = lambda_/(lambda_n_aug);
    for (int i=1; i< n_sig_; i++) {
      weights_(i) = 0.5/(lambda_n_aug);
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];

      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vel_abs = sqrt(pow(rho_dot*cos(phi),2)+pow(rho_dot*sin(phi),2)); 
      float yaw_angle = 0.0;
      float yaw_rate = 0.0;

      x_ << px, py, vel_abs, yaw_angle, yaw_rate;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // initialize state
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];

      x_ << px, py, 0, 0, 0;
    }

    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    // set the time when the state is true
    time_us_ = meas_package.timestamp_;

    return;
  }

  // compute the time elapsed in seconds
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Prediction Step
  Prediction(delta_t);

  // Radar updates
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

  // Laser updates
  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /*** Augmented Sigma Pionts ***/
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  AugmentedSigmaPoints(&Xsig_aug);

  /*** Sigma Points Prediction ***/
  SigmaPointPrediction(Xsig_aug, delta_t);

  /*** State Prediction ***/

  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    NormalizeAngle(&x_diff(3));
    
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // Laser measurement dimension: rho, phi, and rho_dot
  int n_z = 2;

  /*** Sigma Points for Measurement ***/

  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  for (int i = 0; i < n_sig_; i++) {
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }


  /*** Predict mean and covariance matrix ***/

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i <n_sig_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(&z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_laser_;


  /*** General UFK Update ***/

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(&x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Measurement
  VectorXd z = meas_package.raw_measurements_;

  // residual
  VectorXd z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K*S*K.transpose();

  //  calculate the laser NIS
  NIS_laser_ = z.transpose() * S.inverse() * z;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Radar measurement dimension: rho, phi, and rho_dot
  int n_z = 3;

  /*** Sigma Points for Measurement ***/

  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  for (int i = 0; i < n_sig_; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    // transform sigma points into measurement space
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    float rho = sqrt(p_x*p_x + p_y*p_y);
    if (fabs(rho) < EPS) rho = EPS;
    float phi = atan2(p_y,p_x);
    float rho_dot = (p_x*v1 + p_y*v2 ) / rho;

    // measurement model
    Zsig(0,i) = rho;
    Zsig(1,i) = phi;
    Zsig(2,i) = rho_dot;
  }


  /*** Predict mean and covariance matrix ***/

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i <n_sig_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(&z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_radar_;


  /*** General UFK Update ***/

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(&z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(&x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Measurement
  VectorXd z = meas_package.raw_measurements_;

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  NormalizeAngle(&z_diff(1));

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K*S*K.transpose();

  //  calculate the radar NIS
  NIS_radar_ = z.transpose() * S.inverse() * z;
}

/**
 * Normalize an angle to be in the range of [-PI, PI].
 * @param {*double} angle A pointer to the angle to be normalized
 */
void UKF::NormalizeAngle(double *angle) {
  double ang = *angle;
  while (ang >  M_PI) ang -= 2.*M_PI;
  while (ang < -M_PI) ang += 2.*M_PI;
  *angle = ang;
}

/**
 * Compute augmented sigma points and save it to Xsig_aug.
 * @param Xsig_aug A pointer to the matrix for storing sigma pionts
 */
void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {
  // augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  // augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  // sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  // augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  // augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_,n_x_) = pow(std_a_, 2);
  P_aug(n_x_+1,n_x_+1) = pow(std_yawdd_, 2);
  // square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  
  // cache variables for performance speed
  double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
  VectorXd col_var_cache;

  // augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    col_var_cache = sqrt_lambda_n_aug * L.col(i);
    Xsig_aug.col(i+1)        = x_aug + col_var_cache;
    Xsig_aug.col(i+1+n_aug_) = x_aug - col_var_cache;
  }

  // write result
  *Xsig_out = Xsig_aug;
}

/**
 * Predict sigma points, given augmented sigma points.
 * Save sigma points prediction in Xsig_pred_.
 * @param Xsig_aug The augmented sigma points used for prediction
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {
  for (int i = 0; i< n_sig_; i++)
  {
    // extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // cache variables for performance speed
    double sin_yaw = sin(yaw);
    double cos_yaw = cos(yaw);
    double yaw_t2 = yaw + yawd*delta_t;
    double nu_a_delta_t = nu_a * delta_t;
    double half_delta_t = 0.5*delta_t;
    double nu_coef1 = half_delta_t*nu_a_delta_t;
    double nu_coef2 = nu_yawdd*delta_t;

    // avoid division by zero
    if (fabs(yawd) > EPS) {
        double v_yawd = v/yawd;
        px_p = p_x + v_yawd * (sin(yaw_t2) - sin_yaw);
        py_p = p_y + v_yawd * (cos_yaw - cos(yaw_t2));
    }
    else {
        px_p = p_x + v*delta_t*cos_yaw;
        py_p = p_y + v*delta_t*sin_yaw;
    }

    double v_p = v;
    double yaw_p = yaw_t2;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + nu_coef1 * cos_yaw;
    py_p = py_p + nu_coef1 * sin_yaw;
    v_p = v_p + nu_a_delta_t;

    yaw_p = yaw_p + half_delta_t*nu_coef2;
    yawd_p = yawd_p + nu_coef2;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}



