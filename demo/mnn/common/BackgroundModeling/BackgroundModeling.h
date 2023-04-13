#pragma once
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Mat/Mat.h"
#include <memory>
#include <vector>

#define BSJ_AI_BACKGROUND_MODELING_VERSION "v1.0.3.a.20221208"

namespace BSJ_AI {
class BackgroundModeling {
public:
    BackgroundModeling(int nValidityPeriod = 30000, int nSamplingInterval = 1000);

    ~BackgroundModeling();

    int checkForeground(const BSJ_AI::ImageData &data, const std::vector<Rect> &vecObjs, std::vector<float> &vecScores, int nThreshold = 50);

    void getBackgroundImage(BSJ_AI::CV::Mat& matImage);

private:
    int init(int nImgWidth, int nImgHeight, IMAGE_FORMAT format);

    int update(const BSJ_AI::CV::Mat  &matImage);

    void getScore(const BSJ_AI::CV::Mat &matImage, int nThreshold, const std::vector<Rect> &vecObjs, std::vector<float> &vecScores);

    int countMoreThan(const BSJ_AI::CV::Mat &matImg, int nThreshold);

    bool m_bRunningLock;
    bool m_bInitLock;

    int64_t m_nSamplingInterval;
    // int m_nScale;
    int m_nInitCount;
    float m_fAlpha;
    int m_nCount;
    int64_t m_nSamplingTimestamp;
    int m_nSrcWidth;
    int m_nSrcHeight;
    int m_nSrcType;
    BSJ_AI::Size m_szStore;
    BSJ_AI::CV::Mat m_matBackground;
};
} // namespace BSJ_AI