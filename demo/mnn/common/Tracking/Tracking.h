#ifndef _TRACKING_
#define _TRACKING_
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Detector/detector.h"
#include "BSJ_AI_function.h"

#define BSJ_AI_TRACKING_VERSION  "version v1.0.0.a20220923"

namespace BSJ_AI {
class Tracking : public noncopyable_t
{
public:
	struct TrackResult {
		int rId;				// 索引，和输入框vector顺序有关
		int tId;				// 跟踪id
		BSJ_AI::Rect2f pRect;	// 预测框结果
		BSJ_AI::Rect2f tRect;	// 滤波后的目标框
		TrackResult(int _rid = 0, int _tid = 0, BSJ_AI::Rect2f _prect = BSJ_AI::Rect2f(), BSJ_AI::Rect2f _trect = BSJ_AI::Rect2f()) {
			rId		= _rid;
			tId		= _tid;
			pRect	= _prect;
			tRect	= _trect;
		}
	};

public:
	Tracking();
	~Tracking();

 	virtual int track(const std::vector<BSJ_AI::Detector::Object>& vecObjects, std::vector<TrackResult>& trackResults) = 0;

	virtual int getPredict(std::vector<BSJ_AI::Rect2f>& vecRects) = 0;
};
}

#endif // !_TRACKING_
