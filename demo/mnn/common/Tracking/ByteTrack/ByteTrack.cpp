#include "ByteTrack.h"

BSJ_AI::ByteTrack::ByteTrack() {
	m_nFrameId		= 0;
	m_nMaxTimeLost	= 30;
	m_fTrackThresh	= 0.5f;
	m_fHeightThresh = 0.6f;
	m_fMatchThresh	= 0.8f;
}

BSJ_AI::ByteTrack::~ByteTrack() {
}

int BSJ_AI::ByteTrack::track(const std::vector<BSJ_AI::Detector::Object>& vecObjects, std::vector<TrackResult>& trackResults) {
	trackResults.clear();

	//
	std::vector<BSJ_AI::Rect2f> detections;
	std::vector<BSJ_AI::Rect2f> detections_low;
	std::vector<std::vector<double>>    iouMatrix;
	std::vector<int>                    assignment;

	if (vecObjects.size() > 0) {
		for (int i = 0; i < vecObjects.size(); i++) {

		}
	}
	return 0;
}

int BSJ_AI::ByteTrack::getPredict(std::vector<BSJ_AI::Rect2f>& vecRects) {
	return 0;
}
