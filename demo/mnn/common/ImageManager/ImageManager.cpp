#include "ImageManager.h"


ImageManager::ImageManager(int nImgWidthInput, int nImgHeightInput, int nImgKeepType, int nImgKeepNumber, unsigned int nFPS, int nImgWidthKeep, int nImgHeightKeep)
{
	if (nImgWidthInput <= 0 || nImgHeightInput <= 0) {
		LOGE("ImageManager error size of image: [%d,%d].", nImgWidthInput, nImgHeightInput);
		throw "ImageManager error size of image.";
	}
	else {
		m_nImgWidthInput = nImgWidthInput;
		m_nImgHeightInput = nImgHeightInput;
		
		if (nImgWidthKeep <= 0 || nImgHeightKeep <= 0) {
			m_nImgWidthKeep = m_nImgWidthInput;
			m_nImgHeightKeep = m_nImgHeightInput;
		}
		else {
			m_nImgWidthKeep = nImgWidthKeep;
			m_nImgHeightKeep = nImgHeightKeep;
		}
	}

	switch (nImgKeepType) {
	case IMAGEMANAGER_KEEP_TYPE_BGR:
	case IMAGEMANAGER_KEEP_TYPE_GRAY1:
	case IMAGEMANAGER_KEEP_TYPE_GRAY3:
	case IMAGEMANAGER_KEEP_TYPE_GRAY_BGR:
	case IMAGEMANAGER_KEEP_TYPE_YUYV:
	case IMAGEMANAGER_KEEP_TYPE_UYVY:
	case IMAGEMANAGER_KEEP_TYPE_NV21:
	case IMAGEMANAGER_KEEP_TYPE_I420:
    case IMAGEMANAGER_KEEP_TYPE_NV12:
		m_nImgKeepType = nImgKeepType;
		break;
	default:
		LOGE("ImageManager error nImgKeepType: %d.", m_nImgKeepType);
		throw "ImageManager error nImgType.";
	}

	// 图像存储
	m_nImgKeepNumber = MAX(nImgKeepNumber, IMAGEMANAGER_KEEP_MIN_VOLUME);
	for (int i = 0; i < m_nImgKeepNumber; i++) {
		m_vecImages.push_back(cv::Mat::zeros(m_nImgHeightKeep, m_nImgWidthKeep, CV_8UC3));
	}

	// 帧率控制队列
	m_hQueueFPS = new SQ_GF::StatisticalQueue<int>(SQ_GF::QUEUE_FORM::QUEUE_FORM_TICK, 1000, nFPS);
	m_nFPS = nFPS;

	// 调用标识归零
	m_nRepeatFlag = 0;

	this->Reset();
}


ImageManager::~ImageManager()
{
	this->Free();
}


void ImageManager::Free()
{
	for (int i = 0; i < m_vecImages.size(); i++) {
		m_vecImages[i].release();
	}
	m_vecImages.clear();

	if (m_hQueueFPS) {
		delete m_hQueueFPS;
		m_hQueueFPS = 0;
	}
}



void ImageManager::Reset()
{
	m_bImgInputLock = false;
	m_nImgPreNumber = -1;
	m_hQueueFPS->reset();
}