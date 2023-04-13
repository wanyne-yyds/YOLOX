#include "BackgroundModeling.h"
#include <BSJ_AI_config.h>
#include <math.h>

BSJ_AI::BackgroundModeling::BackgroundModeling(int nValidityPeriod, int nSamplingInterval) {
    m_bRunningLock = false;
    m_bInitLock = false;
    m_nSamplingInterval = BSJ_BETWEEN(nSamplingInterval, 100, 5000);
    // m_nScale = 4;
    m_szStore = BSJ_AI::Size(160, 90);
    m_nInitCount = BSJ_MAX(10, 10000 / m_nSamplingInterval);
    //m_fAlpha = 1 - (float)pow(0.1f, m_nSamplingInterval / (float)BSJ_MAX(nValidityPeriod, m_nSamplingInterval));
    m_fAlpha = m_nSamplingInterval / (float)BSJ_MAX(nValidityPeriod, m_nSamplingInterval);

    this->init(1280, 720, IMAGE_FORMAT::BGR888);
}

int BSJ_AI::BackgroundModeling::init(int nImgWidth, int nImgHeight, IMAGE_FORMAT format) {
    if (nImgWidth <= 0 || nImgHeight <= 0) {
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (m_bInitLock) {
        return BSJ_AI_FLAG_INITIALIZATION;
    } else {
        m_bInitLock = true;
    }

    while (m_bRunningLock) {
        BSJ_AI::sleep(100);
    }

    m_nCount = 0;
    m_nSamplingTimestamp = 0;
    m_nSrcWidth = nImgWidth;
    m_nSrcHeight = nImgHeight;
    switch (format) {
    case IMAGE_FORMAT::BGR888:
        m_nSrcType = 3;
        break;
    default:
        break;
    }

    // m_szStore = cv::Size(m_nSrcWidth / m_nScale, m_nSrcHeight / m_nScale);
    m_matBackground = BSJ_AI::CV::Mat(m_szStore.height, m_szStore.width, m_nSrcType);
    m_matBackground.setZeros();

    m_bInitLock = false;

    return 0;
}

BSJ_AI::BackgroundModeling::~BackgroundModeling() {
    m_matBackground.release();
}

int BSJ_AI::BackgroundModeling::update(const BSJ_AI::CV::Mat &matImage) {
    // 时间间隔
    uint64_t tick = BSJ_AI::getTickMillitm();
    // 时间差
    uint64_t diffTick = tick - m_nSamplingTimestamp;
      
    // 首次更新背景要快
    if (m_nCount <= m_nInitCount && diffTick > BSJ_MIN(1000, m_nSamplingInterval)) {
        m_matBackground = m_matBackground + matImage * (1.f / m_nInitCount);
        m_nCount++;
        m_nSamplingTimestamp = tick;
    } else if (diffTick > m_nSamplingInterval) {
        m_matBackground = matImage * m_fAlpha + m_matBackground * (1 - m_fAlpha);
        m_nSamplingTimestamp = tick;
    }

    
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int BSJ_AI::BackgroundModeling::countMoreThan(const BSJ_AI::CV::Mat &matImg, int nThreshold) {
    switch (matImg.channels) {
    case 3: {
        int nCount = 0;
        for (int i = 0; i < matImg.rows; i++) {
            for (int j = 0; j < matImg.cols; j++) {
                int index = (i * matImg.cols + j) * 3;
                /*nCount += (sqrt(pow(matImg.data[index + 0], 2) + pow(matImg.data[index + 1], 2) + pow(matImg.data[index + 2], 2)) / (nThreshold * 3));*/
                nCount += ((matImg.data[index + 0] + matImg.data[index + 1] + matImg.data[index + 2]) / (nThreshold * 3));
            }
        }
        return nCount;
    }
    case 1: {
        // 二值化, 下面做法有问题
        int nCount = 0;
        for (int i = 0; i < matImg.rows; i++) {
            for (int j = 0; j < matImg.cols; j++) {
                int index = i * matImg.cols + j;
                nCount += matImg.data[index];
            }
        }
        return (int)nCount / (matImg.rows * matImg.cols);
    }
    default:
        return -1;
    }

    return 0;
}

void BSJ_AI::BackgroundModeling::getScore(const BSJ_AI::CV::Mat &matImage, int nThreshold, const std::vector<Rect> &vecObjs, std::vector<float> &vecScores) {
    if (vecObjs.size() == 0) {
        return;
    }

    //// 做差求绝对值
    // CV::Mat2f matDiff;
    // cv::absdiff(matImage, m_matBackground, matDiff);	// 省略尺寸相同判断
    //// 二值化
    // cv::Mat matThreshold;
    // int maxval = 255;
    // cv::threshold(matDiff, matThreshold, nThreshold, maxval, CV_THRESH_BINARY);

    vecScores.clear();
    for (std::vector<Rect>::const_iterator it = vecObjs.begin(); it != vecObjs.end(); it++) {
        // 坐标转换
        Rect recRoi;
        recRoi.x = it->x * m_szStore.width / m_nSrcWidth;
        recRoi.y = it->y * m_szStore.height / m_nSrcHeight;
        recRoi.width = it->width * m_szStore.width / m_nSrcWidth;
        recRoi.height = it->height * m_szStore.height / m_nSrcHeight;
        recRoi = recRoi & Rect(0, 0, matImage.cols, matImage.rows);

        int nTotal = recRoi.area();
        if (nTotal == 0) {
            vecScores.push_back(-1.f);
        } else {
            //// 做差求绝对值
            BSJ_AI::CV::Mat matDiff = BSJ_AI::CV::absdiff(matImage(recRoi), m_matBackground(recRoi)); // 省略尺寸相同判断

            // 非零点个数
            int nCount = this->countMoreThan(matDiff, nThreshold);
            vecScores.push_back(nCount / (float)nTotal);
        }
    }
}

int BSJ_AI::BackgroundModeling::checkForeground(const BSJ_AI::ImageData &data, const std::vector<Rect> &vecObjs, std::vector<float> &vecScores, int nThreshold) {
    //
    if (data.imgHeight != m_nSrcHeight || data.imgWidth != m_nSrcWidth) {
        int nResult = this->init(data.imgWidth, data.imgHeight, IMAGE_FORMAT::BGR888);
        if (nResult) {
            LOGE("BSJ_AI::BackgroundModeling::checkForeground err: return %d.\n", nResult);
            return nResult;
        }
    }

    // lock
    if (m_bRunningLock || m_bInitLock) {
        return BSJ_AI_FLAG_BUSY;
    } else {
        m_bRunningLock = true;
    }

    BSJ_AI::CV::Mat matBGR;
    if (data.format == BSJ_AI::IMAGE_FORMAT::NV12) {
        BSJ_AI::CV::Mat matNV12 = BSJ_AI::CV::Mat(data.imgHeight + (data.imgHeight>>1), data.imgWidth, 1, (unsigned char *)data.data);

        BSJ_AI::CV::Mat matImage;
        BSJ_AI::CV::cvtColor(matNV12, matImage, BSJ_AI::CV::COLOR_CONVERT_NV12TOBGR);
        BSJ_AI::CV::resize(matImage, matBGR, m_szStore);
    } else{
        BSJ_AI::CV::Mat matImage = BSJ_AI::CV::Mat(data.imgHeight, data.imgWidth, 3, (unsigned char *)data.data);
        BSJ_AI::CV::resize(matImage, matBGR, m_szStore);
    }
    
    // 更新背景
    this->update(matBGR);

    // 统计得分
    this->getScore(matBGR, nThreshold, vecObjs, vecScores);

    // unlock
    m_bRunningLock = false;

    

    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::BackgroundModeling::getBackgroundImage(BSJ_AI::CV::Mat &matImage) {
    matImage = m_matBackground;
}
