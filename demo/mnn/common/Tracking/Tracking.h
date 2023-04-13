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
		int rId;				// �������������vector˳���й�
		int tId;				// ����id
		BSJ_AI::Rect2f pRect;	// Ԥ�����
		BSJ_AI::Rect2f tRect;	// �˲����Ŀ���
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
