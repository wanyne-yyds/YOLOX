#include "ImageManager.h"
#include <BSJ_CV_define.h>


bool ImageManager::InputBGR(const cv::Mat& matBGR)
{
	if (matBGR.rows != m_nImgHeightKeep || matBGR.cols != m_nImgWidthKeep) {
		BSJ_AI::sleep(1000);
		LOGE("ImageManager::InputBGR err: bad Mat.size(%d,%d)!\n", matBGR.cols, matBGR.rows);
		return false;
	}

	// 新序号
	int number = this->GetNewNumber();
	if (number < 0) {
		BSJ_AI::sleep(1000);
		return false;
	}

	// lock
	if (m_bImgInputLock) {
		// 上锁跳过
		BSJ_AI::sleep(1000);
		return false;
	}
	else {
		m_bImgInputLock = true;
	}

	switch (m_nImgKeepType){
	case IMAGEMANAGER_KEEP_TYPE_BGR:
		m_vecImages[number] = matBGR.clone();
		break;
	case IMAGEMANAGER_KEEP_TYPE_GRAY1:
		cv::cvtColor(matBGR, m_vecImages[number], CV_BGR2GRAY);
		break;
	case IMAGEMANAGER_KEEP_TYPE_GRAY3:
		cv::cvtColor(matBGR, m_vecImages[number], CV_BGR2GRAY);
		cv::cvtColor(m_vecImages[number], m_vecImages[number], CV_GRAY2BGR);
		break;
	case IMAGEMANAGER_KEEP_TYPE_GRAY_BGR:
	{
		cv::Mat matT;
		cv::cvtColor(matBGR, matT, CV_BGR2GRAY);
		cv::resize(matT, matT, cv::Size(matBGR.cols / 3 * 3, matBGR.rows / 3));
		m_vecImages[number] = cv::Mat(matBGR.rows / 3, matBGR.cols / 3, CV_8UC3);
		memcpy(m_vecImages[number].data, matT.data, matT.rows * matT.cols);
		matT.release();
	}
	break;
	case IMAGEMANAGER_KEEP_TYPE_YUYV:
	case IMAGEMANAGER_KEEP_TYPE_UYVY:
	case IMAGEMANAGER_KEEP_TYPE_NV21:
	case IMAGEMANAGER_KEEP_TYPE_I420:
	default:
		BSJ_AI::sleep(1000);
		LOGE("ImageManager::InputBGR err: bad type(%d)!\n", m_nImgKeepType);
		m_bImgInputLock = false;
		return false;
	}

	// 更新序号
	m_nImgPreNumber = number;

	// 更新帧率
	if (m_hQueueFPS) {
		m_hQueueFPS->push(1);
	}

	// 调用标识归零
	m_nRepeatFlag = 0;

	// unlock
	m_bImgInputLock = false;

	return true;
}