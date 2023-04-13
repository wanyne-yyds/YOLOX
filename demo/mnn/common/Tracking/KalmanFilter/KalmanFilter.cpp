#include "KalmanFilter.h"

BSJ_AI::KalmanFilter::KalmanFilter()
{
}

BSJ_AI::KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams)
{
    init(dynamParams, measureParams, controlParams);
}

BSJ_AI::KalmanFilter::~KalmanFilter()
{
}

void BSJ_AI::KalmanFilter::init(int dynamParams, int measureParams, int controlParams)
{
    controlParams = BSJ_MAX(controlParams, 0);

    statePre = BSJ_AI::CV::Mat2f(dynamParams, 1);  statePre.setZeros();
    statePost = BSJ_AI::CV::Mat2f(dynamParams, 1); statePost.setZeros();
    transitionMatrix = BSJ_AI::CV::Mat2f(dynamParams, dynamParams); transitionMatrix.setEye();
    
    processNoiseCov = BSJ_AI::CV::Mat2f(dynamParams, dynamParams); processNoiseCov.setEye();
    measurementMatrix = BSJ_AI::CV::Mat2f(measureParams, dynamParams);  measurementMatrix.setZeros();
    measurementNoiseCov = BSJ_AI::CV::Mat2f(measureParams, measureParams); measurementNoiseCov.setEye();

    errorCovPre = BSJ_AI::CV::Mat2f(dynamParams, dynamParams); errorCovPre.setZeros();
    errorCovPost = BSJ_AI::CV::Mat2f(dynamParams, dynamParams);  errorCovPost.setZeros();
    gain = BSJ_AI::CV::Mat2f(dynamParams, measureParams); gain.setZeros();

    if (controlParams > 0) {
        controlMatrix = BSJ_AI::CV::Mat2f(dynamParams, controlParams);
        controlMatrix.setZeros();
    } else {
        controlMatrix.release();
    }

    temp1 = BSJ_AI::CV::Mat2f(dynamParams, dynamParams);
    temp2 = BSJ_AI::CV::Mat2f(measureParams, dynamParams);
    temp3 = BSJ_AI::CV::Mat2f(measureParams, measureParams);
    temp4 = BSJ_AI::CV::Mat2f(measureParams, dynamParams);
    temp5 = BSJ_AI::CV::Mat2f(measureParams, 1);
}

const BSJ_AI::CV::Mat2f& BSJ_AI::KalmanFilter::predict(const BSJ_AI::CV::Mat2f& control)
{
    // update the state: x'(k) = A*x(k)
    statePre = transitionMatrix * statePost;

    if (!control.empty())
    {
        // x'(k) = x'(k) + B*u(k)
        statePre += controlMatrix * control;
    }

    // update error covariance matrices: temp1 = A*P(k)
    temp1 = transitionMatrix * errorCovPost;

    // P'(k) = temp1*At + Q
    errorCovPre = temp1 * transitionMatrix.t() + processNoiseCov;

    // handle the case when there will be measurement before the next predict.
    statePost = statePre;
    errorCovPost = errorCovPre;

    return statePre;
}

const BSJ_AI::CV::Mat2f& BSJ_AI::KalmanFilter::correct(const BSJ_AI::CV::Mat2f& measurement)
{
    // temp2 = H*P'(k)
    temp2 = measurementMatrix * errorCovPre;

    // temp3 = temp2*Ht + R
    temp3 = temp2 * measurementMatrix.t() + measurementNoiseCov;

    // temp4 = inv(temp3)*temp2 = Kt(k)
    temp4 = temp3.inv() * temp2;

    // K(k)
    gain = temp4.t();

    // temp5 = z(k) - H*x'(k)
    temp5 = measurement - measurementMatrix * statePre;

    // x(k) = x'(k) + K(k)*temp5
    statePost = statePre + gain * temp5;

    // P(k) = P'(k) - K(k)*temp2
    errorCovPost = errorCovPre - gain * temp2;

    return statePost;
}


