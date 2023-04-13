#ifndef _BYTE_TRACK_
#define _BYTE_TRACK_
#include <memory>
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Mat/Mat.h"
#include "../Tracking.h"
#include "../KalmanTracker/KalmanTracker.h"

namespace BSJ_AI {
class ByteTrack : public Tracking {
public:
	ByteTrack();
	~ByteTrack();

	int track(const std::vector<BSJ_AI::Detector::Object>& vecObjects, std::vector<TrackResult>& trackResults);

	int getPredict(std::vector<BSJ_AI::Rect2f>& vecRects);
private:
	int		m_nFrameId;
	int		m_nMaxTimeLost;
	float	m_fTrackThresh;
	float	m_fHeightThresh;
	float	m_fMatchThresh;
};
}

#endif // !_BYTE_TRACK_
