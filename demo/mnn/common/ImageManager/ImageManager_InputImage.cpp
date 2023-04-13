#include "ImageManager.h"
#include <BSJ_CV_define.h>


int ImageManager::InputImage(int dataLength, char* data, int imgHeight, int imgWidth, int format)
{
	// 参数判断
	if (data == NULL) {
		BSJ_AI::sleep(100);
		LOGI("ImageManager::InputImage err: data is NULL!\n");
		return IMAGEMANAGER_FLAG_NULL_POINTER;
	}
	else if (imgHeight != m_nImgHeightInput || imgWidth != m_nImgWidthInput) {
		BSJ_AI::sleep(100);
		LOGE("ImageManager::InputImage err: bad size(%d,%d)!\n", imgWidth, imgHeight);
		return IMAGEMANAGER_FLAG_BAD_PARAMETER;
	}

	// 限定帧率
	if (!this->KeepNecessary()) {
		BSJ_AI::sleep(10);
		LOGI("ImageManager::InputImage err: InputImage no necessary!\n");
		return IMAGEMANAGER_FLAG_NO_NECESSARY;
	}

	// 判断数据长度
	switch (format) {
	case IMAGEMANAGER_FORMAT_YUYV:
	case IMAGEMANAGER_FORMAT_UYVY:
	case IMAGEMANAGER_FORMAT_RGB565:
	case IMAGEMANAGER_FORMAT_BGR565:
		if (imgHeight * imgWidth * 2 != dataLength) {
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage err: bad length of data, format=%d, height * width * 2 != dataLength, dataLength=%d, size(%d,%d)", format, dataLength, imgWidth, imgHeight);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
		break;
	case IMAGEMANAGER_FORMAT_NV21:
	case IMAGEMANAGER_FORMAT_NV12:
	case IMAGEMANAGER_FORMAT_I420:
		if (imgHeight * imgWidth * 3 / 2 != dataLength) {
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage err: bad length of data, format=%d, height * width * 3 / 2 != dataLength, dataLength=%d, size(%d,%d)", format, dataLength, imgWidth, imgHeight);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
		break;
	case IMAGEMANAGER_FORMAT_RGB888:
	case IMAGEMANAGER_FORMAT_BGR888:
		if (imgHeight * imgWidth * 3 != dataLength) {
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage err: bad length of data, format=%d, height * width * 3 != dataLength, dataLength=%d, size(%d,%d)", format, dataLength, imgWidth, imgHeight);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
		break;
	default:
		BSJ_AI::sleep(100);
		LOGE("ImageManager::InputImage err: format: %d.\n", format);
		return IMAGEMANAGER_FLAG_BAD_PARAMETER;
	}

	// 新序号
	int number = this->GetNewNumber();
	if (number < 0) {
		BSJ_AI::sleep(10);
		LOGE("ImageManager::InputImage err: GetNewNumber failed!");
		return IMAGEMANAGER_FLAG_FULL_QUEUE;
	}

	// lock
	if (m_bImgInputLock) {
		// 上锁跳过
		BSJ_AI::sleep(10);
		return IMAGEMANAGER_FLAG_BUSY;
	}
	else {
		m_bImgInputLock = true;
	}

	// 格式转换
	cv::Mat matData;
	cv::Mat matGray;
	bool bResult = true;
	if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_BGR) {
		switch (format) {
		case IMAGEMANAGER_FORMAT_YUYV:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2BGR_YUYV);
			break;
		case IMAGEMANAGER_FORMAT_UYVY:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2BGR_UYVY);
			break;
		case IMAGEMANAGER_FORMAT_NV21:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV420sp2BGR);
			break;
		case IMAGEMANAGER_FORMAT_I420:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2BGR_I420);
			break;
		case IMAGEMANAGER_FORMAT_RGB565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_BGR5652RGB);
			break;
		case IMAGEMANAGER_FORMAT_BGR565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_BGR5652BGR);
			break;
		case IMAGEMANAGER_FORMAT_RGB888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, m_vecImages[number], CV_BGR2RGB);
			break;
		case IMAGEMANAGER_FORMAT_BGR888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			m_vecImages[number] = matData.clone();
			break;
		case IMAGEMANAGER_FORMAT_NV12:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2BGR_NV12);
			break;
		default:
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage error format: %d.\n", format);
			bResult = false;
		}

		if (bResult) {
			// resize
			if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
				cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidthKeep, m_nImgHeightKeep));
			}
		}
	}
	else if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_GRAY1) {
		switch (format) {
		case IMAGEMANAGER_FORMAT_YUYV:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_YUYV);
			break;
		case IMAGEMANAGER_FORMAT_UYVY:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_UYVY);
			break;
		case IMAGEMANAGER_FORMAT_NV21:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV420sp2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_I420:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_I420);
			break;
		case IMAGEMANAGER_FORMAT_RGB565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, matData, CV_BGR5652BGR);
			cvtColor(matData, m_vecImages[number], CV_RGB2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_BGR565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_BGR5652GRAY);
			break;
		case IMAGEMANAGER_FORMAT_RGB888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, m_vecImages[number], CV_RGB2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_BGR888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, m_vecImages[number], CV_BGR2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_NV12:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_NV12);
			break;
		default:
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage error format: %d.\n", format);
			bResult = false;
		}
		
		if (bResult) {
			// resize
			if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
				cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidthKeep, m_nImgHeightKeep));
			}
		}
	}
	else if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_GRAY3) {
		switch (format) {
		case IMAGEMANAGER_FORMAT_YUYV:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_YUYV);
			break;
		case IMAGEMANAGER_FORMAT_UYVY:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_UYVY);
			break;
		case IMAGEMANAGER_FORMAT_NV21:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV420sp2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_I420:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_I420);
			break;
		case IMAGEMANAGER_FORMAT_RGB565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, matData, CV_BGR5652BGR);
			cvtColor(matData, m_vecImages[number], CV_RGB2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_BGR565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, m_vecImages[number], CV_BGR5652GRAY);
			break;
		case IMAGEMANAGER_FORMAT_RGB888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, m_vecImages[number], CV_RGB2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_BGR888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, m_vecImages[number], CV_BGR2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_NV12:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, m_vecImages[number], CV_YUV2GRAY_NV12);
			break;
		default:
			bResult = false;
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage error format: %d.\n", format);
		}

		if (bResult) {
			// resize
			if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
				cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidthKeep, m_nImgHeightKeep));
			}
			cvtColor(m_vecImages[number], m_vecImages[number], CV_GRAY2BGR);
		}
	}
	else if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_GRAY_BGR) {
		switch (format) {
		case IMAGEMANAGER_FORMAT_YUYV:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, matGray, CV_YUV2GRAY_YUYV);
			break;
		case IMAGEMANAGER_FORMAT_UYVY:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, matGray, CV_YUV2GRAY_UYVY);
			break;
		case IMAGEMANAGER_FORMAT_NV21:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, matGray, CV_YUV420sp2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_I420:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, matGray, CV_YUV2GRAY_I420);
			break;
		case IMAGEMANAGER_FORMAT_RGB565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, matData, CV_BGR5652BGR);
			cvtColor(matData, matGray, CV_RGB2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_BGR565:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			cvtColor(matData, matGray, CV_BGR5652GRAY);
			break;
		case IMAGEMANAGER_FORMAT_RGB888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, matGray, CV_RGB2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_BGR888:
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC3, data);
			cvtColor(matData, matGray, CV_BGR2GRAY);
			break;
		case IMAGEMANAGER_FORMAT_NV12:
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			cvtColor(matData, matGray, CV_YUV2GRAY_NV12);
			break;
		default:
			bResult = false;
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage error format: %d.\n", format);
		}

		if (bResult) {
			cv::resize(matGray, matGray, cv::Size(m_nImgWidthKeep / 3 * 3, m_nImgHeightKeep / 3));
			m_vecImages[number] = cv::Mat(m_nImgHeightKeep / 3, m_nImgWidthKeep / 3, CV_8UC3);
			memcpy(m_vecImages[number].data, matGray.data, matGray.rows * matGray.cols);
		}
	}
	else {
		if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage err: bad size(%d,%d)! keepType = %d.\n", imgWidth, imgHeight, m_nImgKeepType);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}

		if ((m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_YUYV && format == IMAGEMANAGER_FORMAT_YUYV)
			|| (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_UYVY && format == IMAGEMANAGER_FORMAT_UYVY)) {
			matData = cv::Mat(imgHeight, imgWidth, CV_8UC2, data);
			m_vecImages[number] = matData.clone();
		}
		else if ((m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_NV21 && format == IMAGEMANAGER_FORMAT_NV21)
			|| (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_I420 && format == IMAGEMANAGER_FORMAT_I420)
                   || (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_NV12 && format == IMAGEMANAGER_FORMAT_NV12)) {
			matData = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1, data);
			m_vecImages[number] = matData.clone();
		}
		else {
			bResult = false;
			BSJ_AI::sleep(100);
			LOGE("ImageManager::InputImage err: bad format(%d).\n", format);
		}
	}
	matData.release();
	matGray.release();
	
	if (bResult) {
		// 更新序号
		m_nImgPreNumber = number;
		// 更新帧率
		if (m_hQueueFPS) {
			m_hQueueFPS->push(1);
		}
		// 调用标识归零
		m_nRepeatFlag = 0;
	}

	// unlock
	m_bImgInputLock = false;

	return IMAGEMANAGER_FLAG_SUCCESSFUL;
}


int GetRefcount(const cv::Mat& mat)
{
#if BSJ_CV_VERSION == 2
	if (mat.refcount == nullptr) {
		return 0;
	}
	else {
		return (*(mat.refcount));
	}
#else
	return (mat.u ? (mat.u->refcount) : 0);
#endif
}

int ImageManager::GetNewNumber()
{
	for (int i = 1; i < m_nImgKeepNumber; i++) {
		int number = (m_nImgPreNumber + i) % m_nImgKeepNumber;
		switch (GetRefcount(m_vecImages[number])) {
		case 0:
			return number;
		case 1:
			//m_vecImages[number].release();
			return number;
		}
	}

	return -1;
}