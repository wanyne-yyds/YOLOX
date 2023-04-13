#include "ImageManager.h"


bool ImageManager::KeepNecessary()
{
	if (!m_hQueueFPS) {
		LOGE("ImageManager::KeepNecessary err: m_hQueueFPS is NULL!\n");
		return false;
	}
	
	int nResult = m_hQueueFPS->push(0);
	if (nResult == SQ_FLAG_MORE_THAN) {
		return false;
	}
	else {
		int nMean = m_hQueueFPS->getMean();
		return (nMean <= m_nFPS);
	}
}