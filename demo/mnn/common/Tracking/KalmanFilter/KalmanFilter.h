#ifndef _KALMAN_FILTER_H_
#define _KALMAN_FILTER_H_

#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Mat/Mat.h"
// #include <opencv2/opencv.hpp>

namespace BSJ_AI
{
class KalmanFilter
{
public:
	KalmanFilter();
	KalmanFilter(int dynamParams, int measureParams, int controlParams = 0);
	~KalmanFilter();

	const BSJ_AI::CV::Mat2f& predict(const BSJ_AI::CV::Mat2f& control = BSJ_AI::CV::Mat2f());

    const BSJ_AI::CV::Mat2f& correct(const BSJ_AI::CV::Mat2f& measurement);

    BSJ_AI::CV::Mat2f statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    BSJ_AI::CV::Mat2f statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    BSJ_AI::CV::Mat2f transitionMatrix;   //!< state transition matrix (A)
    BSJ_AI::CV::Mat2f controlMatrix;      //!< control matrix (B) (not used if there is no control)
    BSJ_AI::CV::Mat2f measurementMatrix;  //!< measurement matrix (H)
    BSJ_AI::CV::Mat2f processNoiseCov;    //!< process noise covariance matrix (Q)
    BSJ_AI::CV::Mat2f measurementNoiseCov;//!< measurement noise covariance matrix (R)
    BSJ_AI::CV::Mat2f errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
    BSJ_AI::CV::Mat2f gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
    BSJ_AI::CV::Mat2f errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)

private:
	void init(int dynamParams, int measureParams, int controlParams = 0);

    // temporary matrices
    BSJ_AI::CV::Mat2f temp1;
    BSJ_AI::CV::Mat2f temp2;
    BSJ_AI::CV::Mat2f temp3;
    BSJ_AI::CV::Mat2f temp4;
    BSJ_AI::CV::Mat2f temp5;

};
}


#endif // !_KALMAN_H_