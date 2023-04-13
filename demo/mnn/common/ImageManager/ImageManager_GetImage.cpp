#include "ImageManager.h"
#include <time.h>


bool ImageManager::GetImage(cv::Mat& matImage, int nGetID)
{
	if (m_nImgPreNumber < 0) {
		LOGE("ImageManager::GetImage err: m_nImgPreNumber = %d < 0\n", m_nImgPreNumber);
		BSJ_AI::sleep(1000);
		return false;
	}
	else if (m_nImgPreNumber >= m_vecImages.size()) {
		LOGE("ImageManager::GetImage err: m_nImgPreNumber = %d >= m_vecImages.size() = %zd\n", m_nImgPreNumber, m_vecImages.size());
		BSJ_AI::sleep(1000);
		return false;
	}

	if (nGetID >= 0) {
		if ((1 << nGetID) & m_nRepeatFlag) {
			LOGI("ImageManager::GetImage info: wait for fresh image.\n");
			BSJ_AI::sleep(10);
			return false;
		}
		else {
			m_nRepeatFlag |= (1 << nGetID);
			matImage = m_vecImages[m_nImgPreNumber];
			return true;
		}
	}
	else {
		// nGetID < 0��������ȡͼ
		matImage = m_vecImages[m_nImgPreNumber];
		return true;
	}
}


void ImageManager::SaveImage(const std::string& strPath)
{
	char fileName[1024];

	// ����ͼ��
	time_t nowtime;
	time(&nowtime);
	struct tm* info = localtime(&nowtime);

	for (int i = 0; i < m_nImgKeepNumber; i++) {
		sprintf(fileName, "%s/SaveImage%04d%02d%02d%02d%02d%02d_%d_%d.jpg",
			strPath.c_str(), info->tm_year + 1900, info->tm_mon + 1, info->tm_mday, info->tm_hour, info->tm_min, info->tm_sec, m_nImgPreNumber, i);
		cv::imwrite(fileName, m_vecImages[i]);
	}
}