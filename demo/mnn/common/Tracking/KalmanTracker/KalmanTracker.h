#pragma once
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Tracking/KalmanFilter/KalmanFilter.h"
#include "Mat/Mat.h"
//#include <opencv2/opencv.hpp>

namespace BSJ_AI
{
class KalmanTracker
{
public:
	KalmanTracker();

	KalmanTracker(BSJ_AI::Rect2f initRect);

	~KalmanTracker();
	
	BSJ_AI::Rect2f predict();
	void update(BSJ_AI::Rect2f stateMat);

	BSJ_AI::Rect2f get_state();
	BSJ_AI::Rect2f get_rect_xysr(float cx, float cy, float s, float r);

	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;

private:
	void init_kf(BSJ_AI::Rect2f stateMat);

	BSJ_AI::KalmanFilter kf;
	BSJ_AI::CV::Mat2f measurement;

	//cv::KalmanFilter kf1;
	//cv::Mat measurement1;

	std::vector<BSJ_AI::Rect2f> m_history;
};
}