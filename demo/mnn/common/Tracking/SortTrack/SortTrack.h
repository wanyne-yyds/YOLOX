#ifndef _SORT_TRACK_
#define _SORT_TRACK_
#include <memory>
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Mat/Mat.h"
#include "../KalmanTracker/KalmanTracker.h"

namespace BSJ_AI {
class SortTrack {
public:
	SortTrack();
	~SortTrack();

	struct TrackResult {
		int rId;	// �������������vector˳���й�
		int tId;	// ����id
		BSJ_AI::Rect2f pRect;	// Ԥ�����
		BSJ_AI::Rect2f tRect;	// �˲����Ŀ���
		TrackResult(int _rid = 0, int _tid=0, BSJ_AI::Rect2f _prect= BSJ_AI::Rect2f(), BSJ_AI::Rect2f _trect= BSJ_AI::Rect2f()) {
			rId = _rid;
			tId = _tid;
			pRect = _prect;
			tRect = _trect;
		}
	};

	int track(const std::vector<BSJ_AI::Rect2f>& vecRects, std::vector<TrackResult>& trackResults, std::vector<BSJ_AI::Rect2f>& vecPredRects);

private:
	std::vector<std::shared_ptr<BSJ_AI::KalmanTracker>>	m_vhTrackers;

	double GetIOU(BSJ_AI::Rect2f bb_dr, BSJ_AI::Rect2f bb_gt);
	
	unsigned int m_trackCountId;

private:

};
}
#endif // !_SORT_TRACK_