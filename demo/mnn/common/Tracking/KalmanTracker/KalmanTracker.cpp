#include "KalmanTracker.h"
#include <cmath>

BSJ_AI::KalmanTracker::KalmanTracker()
{
	init_kf(BSJ_AI::Rect2f());
	m_time_since_update = 0;
	m_hits = 0;
	m_hit_streak = 0;
	m_age = 0;
	m_id = 0;
}

BSJ_AI::KalmanTracker::KalmanTracker(BSJ_AI::Rect2f initRect)
{
	init_kf(initRect);
	m_time_since_update = 0;
	m_hits = 0;
	m_hit_streak = 0;
	m_age = 0;
	m_id = 0;
}

BSJ_AI::KalmanTracker::~KalmanTracker()
{
	m_history.clear();
}

void BSJ_AI::KalmanTracker::init_kf(BSJ_AI::Rect2f stateMat)
{
	int stateNum = 7;
	int measureNum = 4;
	kf = BSJ_AI::KalmanFilter(stateNum, measureNum, 0);
	//// opencv
	//kf1 = cv::KalmanFilter(stateNum, measureNum, 0);

	//measurement1 = cv::Mat::zeros(measureNum, 1, CV_32F);

	//kf1.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
	//	1, 0, 0, 0, 1, 0, 0,
	//	0, 1, 0, 0, 0, 1, 0,
	//	0, 0, 1, 0, 0, 0, 1,
	//	0, 0, 0, 1, 0, 0, 0,
	//	0, 0, 0, 0, 1, 0, 0,
	//	0, 0, 0, 0, 0, 1, 0,
	//	0, 0, 0, 0, 0, 0, 1);

	//cv::setIdentity(kf1.measurementMatrix);
	//cv::setIdentity(kf1.processNoiseCov, cv::Scalar::all(1e-2));
	//cv::setIdentity(kf1.measurementNoiseCov, cv::Scalar::all(1e-1));
	//cv::setIdentity(kf1.errorCovPost, cv::Scalar::all(1));

	//// initialize state vector with bounding box in [cx,cy,s,r] style
	//kf1.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	//kf1.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	//kf1.statePost.at<float>(2, 0) = stateMat.area();
	//kf1.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
	////opencv

	measurement = BSJ_AI::CV::Mat2f(measureNum, 1);
	measurement.setZeros();

	kf.transitionMatrix = BSJ_AI::CV::Mat2f(stateNum, stateNum);
	std::vector<float> data{
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1 };
	::memcpy(kf.transitionMatrix.data, (float*)data.data(), data.size() * sizeof(float));

	kf.measurementMatrix.setEye();
	kf.processNoiseCov.setEye();
	kf.processNoiseCov = kf.processNoiseCov * 1e-2;

	kf.measurementNoiseCov.setEye();
	kf.measurementNoiseCov = kf.measurementNoiseCov * 1e-1;

	kf.errorCovPost.setEye();

	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.data[0 * kf.statePost.cols] = stateMat.x + stateMat.width / 2;
	kf.statePost.data[1 * kf.statePost.cols] = stateMat.y + stateMat.height / 2;
	kf.statePost.data[2 * kf.statePost.cols] = stateMat.area();
	kf.statePost.data[3 * kf.statePost.cols] = stateMat.width / stateMat.height;
}

BSJ_AI::Rect2f BSJ_AI::KalmanTracker::predict()
{
	// predict
	BSJ_AI::CV::Mat2f p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	BSJ_AI::Rect2f predictBox = get_rect_xysr(p.data[0 * p.cols], p.data[1 * p.cols], p.data[2 * p.cols], p.data[3 * p.cols]);

	m_history.push_back(predictBox);
	return m_history.back();

	//// predict
	//cv::Mat p1 = kf1.predict();
	//m_age += 1;

	//if (m_time_since_update > 0)
	//	m_hit_streak = 0;
	//m_time_since_update += 1;

	//BSJ_AI::Rect2f predictBox = get_rect_xysr(p1.at<float>(0, 0), p1.at<float>(1, 0), p1.at<float>(2, 0), p1.at<float>(3, 0));

	//m_history.push_back(predictBox);
	//return m_history.back();
}

void BSJ_AI::KalmanTracker::update(BSJ_AI::Rect2f stateMat)
{
	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;

	// measurement
	measurement.data[0 * measurement.cols] = stateMat.x + stateMat.width / 2;
	measurement.data[1 * measurement.cols] = stateMat.y + stateMat.height / 2;
	measurement.data[2 * measurement.cols] = stateMat.area();
	measurement.data[3 * measurement.cols] = stateMat.width / stateMat.height;

	// update
	kf.correct(measurement);

	//m_time_since_update = 0;
	//m_history.clear();
	//m_hits += 1;
	//m_hit_streak += 1;

	//// measurement
	//measurement1.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	//measurement1.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	//measurement1.at<float>(2, 0) = stateMat.area();
	//measurement1.at<float>(3, 0) = stateMat.width / stateMat.height;

	//// update
	//kf1.correct(measurement1);
}

BSJ_AI::Rect2f BSJ_AI::KalmanTracker::get_state()
{
	BSJ_AI::CV::Mat2f s = kf.statePost;
	return get_rect_xysr(s.data[0 * s.cols], s.data[1 * s.cols], s.data[2 * s.cols], s.data[3 * s.cols]);
	// cv::Mat s1 = kf1.statePost;
	// return get_rect_xysr(s1.at<float>(0, 0), s1.at<float>(1, 0), s1.at<float>(2, 0), s1.at<float>(3, 0));
}

BSJ_AI::Rect2f BSJ_AI::KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = std::sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;
	return BSJ_AI::Rect2f(x, y, w, h);
}
