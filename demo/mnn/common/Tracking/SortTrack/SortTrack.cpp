#include "SortTrack.h"
#include "../Hungarian/Hungarian.h"
#include <set>
#include <climits>
BSJ_AI::SortTrack::SortTrack()
{
    m_trackCountId = 0;
}

BSJ_AI::SortTrack::~SortTrack()
{
}

int BSJ_AI::SortTrack::track(const std::vector<BSJ_AI::Rect2f>& vecRects, std::vector<TrackResult>& trackResults, std::vector<BSJ_AI::Rect2f>& vecPredRects)
//int BSJ_AI::SortTrack::Sortx(std::vector<BSJ_AI::Rect> vecRects, std::vector<TrackingRect>& vecTrackRects, std::vector<BSJ_AI::Rect2f>& vecPredRects)
{
    trackResults.clear();
    vecPredRects.clear();

    int max_age         = 90;   // ����Ŀ������ٴ���
    int min_hits        = 3;    // Ŀ����ʧ����С֡��
    double iouThreshold = 0.3;  // ƥ��iou

    // variables used in the sort-loop
    std::vector<BSJ_AI::Rect2f>         predictedBoxes;
    std::vector<std::vector<double>>    iouMatrix;
    std::vector<int>                    assignment;

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;

    // result
    std::vector<BSJ_AI::Point> matchedPairs;

    unsigned int trkNum = 0;
    unsigned int detNum = 0;


    // �״γ�ʼ���������˲�
    if (m_vhTrackers.size() == 0) {
        // ��ʼ�� & ���ص�һ�θ��ٽ��
        for (unsigned int i = 0; i < vecRects.size(); i++) {
            // ��ʼ��
            m_vhTrackers.emplace_back(
                new BSJ_AI::KalmanTracker(vecRects[i])
            );

            m_vhTrackers[int(m_vhTrackers.size()) - 1]->m_id = m_trackCountId++;
            // ���ص�һ�θ��ٽ��
            trackResults.push_back(TrackResult(i, i+1, vecRects[i], BSJ_AI::Rect2f()));
        }
        return BSJ_AI_FLAG_SUCCESSFUL;
    }

    /*
   3.1. �������˲���ȡ��ǰ֡�ĸ��ٽ��
   */
    for (auto it = m_vhTrackers.begin(); it != m_vhTrackers.end();) {
        BSJ_AI::Rect2f pBox = (*it)->predict();
        if (pBox.x >= 0 && pBox.y >= 0) {
            predictedBoxes.push_back(pBox);
            it++;
        } else {
            it = m_vhTrackers.erase(it);
        }
    }

    vecPredRects = predictedBoxes;

    /*
    3.2. ����������Ϳ���Ԥ����
    */
    trkNum = predictedBoxes.size();
    detNum = vecRects.size();
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

    // ���������Ϳ���Ԥ������iou
    for (unsigned int i = 0; i < trkNum; i++) {
        for (unsigned int j = 0; j < detNum; j++) {
            // ʹ��1-iou����Ϊ�������㷨���������С�ɱ����䡣
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], vecRects[j]);
        }
    }

    // ʹ���������㷨����������
    // ���ؽ���ǿ����˲����ĳ��� [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(iouMatrix, assignment);

    // ����ƥ��, unmatched_detections and unmatched_predictions
    //	��ƥ��Ŀ��
    if (detNum > trkNum)  {
        for (unsigned int n = 0; n < detNum; n++) {
            allItems.insert(n);
        }

        for (unsigned int i = 0; i < trkNum; ++i) {
            matchedItems.insert(assignment[i]);
        }

        // ����allItems��matchedItems֮��Ĳ�ֵ������unmatchedDetections
        std::set_difference(allItems.begin(), allItems.end(),
            matchedItems.begin(), matchedItems.end(),
            insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) {// there are unmatched trajectory/predictions
        for (unsigned int i = 0; i < trkNum; ++i) {
            // unassigned label will be set as -1 in the assignment algorithm
            if (assignment[i] == -1) {
                unmatchedTrajectories.insert(i);
            }
        }
    } else {

    }

    // filter out matched with low IOU
    // output matchedPairs
    for (unsigned int i = 0; i < trkNum; ++i) {
        // pass over invalid values
        if (assignment[i] == -1) continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else {
            matchedPairs.push_back(BSJ_AI::Point(i, assignment[i]));
        }
    }

    /*
    3.3. updating trackers
    update matched trackers with assigned detections.
    each prediction is corresponding to a tracker
    */
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++) {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        m_vhTrackers[trkIdx]->update(vecRects[detIdx]);

        if ((m_vhTrackers[trkIdx]->m_time_since_update < 1) && (m_vhTrackers[trkIdx]->m_hit_streak >= min_hits)) {
            trackResults.push_back(TrackResult(detIdx, m_vhTrackers[trkIdx]->m_id + 1, vecRects[detIdx], m_vhTrackers[trkIdx]->get_state()));
        }

    }

    // create and initialize new trackers for unmatched detections
    for (auto umd : unmatchedDetections) {
        //KalmanTracker tracker = KalmanTracker(detData[umd].box);
        //trackers.push_back(tracker);
        // ��ʼ��
        m_vhTrackers.emplace_back(new BSJ_AI::KalmanTracker(vecRects[umd]));
        m_vhTrackers[int(m_vhTrackers.size()) - 1]->m_id = m_trackCountId++;

        if ((m_vhTrackers[umd]->m_time_since_update < 1) && (m_vhTrackers[umd]->m_hit_streak >= min_hits)) {
            trackResults.push_back(TrackResult(umd, m_vhTrackers[umd]->m_id + 1, vecRects[umd], m_vhTrackers[umd]->get_state()));
        }

    }

    // get trackers' output
    for (auto it = m_vhTrackers.begin(); it != m_vhTrackers.end();) {        
        it++;

        // remove dead tracklet
        if (it != m_vhTrackers.end() && (*it)->m_time_since_update > max_age) {
            it = m_vhTrackers.erase(it);
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

double BSJ_AI::SortTrack::GetIOU(BSJ_AI::Rect2f bb_dr, BSJ_AI::Rect2f bb_gt)
{
    float in = (bb_dr & bb_gt).area();
    float un = bb_dr.area() + bb_gt.area() - in;

    if (un < 0.001f)
        return 0;

    double iou = in / un;

    return iou;
}
