// CameraCoords.cpp : 定义 DLL 应用程序的导出函数。
//

#include "CameraCoords.h"
#include <stdlib.h>
#include "math.h"
#include <string.h>

bool CameraCoords::InitCoords()		
{
	int nFocus = m_nFocus;
	int nH = m_nH;
	float fTheta = m_fTheta;

	m_anDistance = (int*)malloc(m_nImgHeight * sizeof(int));
	m_afWidth = (float*)malloc(m_nImgHeight * sizeof(float));
	// out of memory
	if (!m_anDistance || !m_afWidth) {
		m_nImgHeight = 0;
		m_nImgWidth = 0;
		return false;
	}
	memset(m_anDistance, 0x00, m_nImgHeight * sizeof(int));
	memset(m_afWidth, 0x00, m_nImgHeight * sizeof(int));

	float y0 = (m_nImgHeight + 1) * 0.5f;		// 水平中轴

	float *h = (float*)malloc(m_nImgHeight * sizeof(float));
	if (!h) {
		return false;
	}
	float *k = (float*)malloc(m_nImgHeight * sizeof(float));
	if (!k) {
		free(h);
		return false;
	}

	float alpha;
	
	float costheta1 = 1 / cos(fTheta);
	float tantheta = tan(fTheta);
	float htan = nH * tantheta;
	float fcos = nFocus * costheta1;
	float hcos = nH * costheta1;
	float ftan = nFocus * tantheta;
	float f1 = 1 / (float)nFocus;


	for (int i = 0; i<y0; i++) {
		h[i] = (i + 1 - y0);
		h[m_nImgHeight - i - 1] = -h[i];

		alpha = atan(h[i] * f1);
		float t = cos(alpha);
		k[i] = t / cos(fTheta + alpha);
		k[m_nImgHeight - i - 1] = t / cos(fTheta - alpha);
	}

	// 计算距离、高度、宽度
	for (unsigned int j = m_nTopValid; j<m_nImgHeight; j++) {
		float t = hcos / (h[j] - ftan);
		m_anDistance[j] = (int)(htan + fcos * t);
		m_afWidth[j] = k[j] * t;
	}

	free(h);
	free(k);

	// 脚底图像位置		d = htan + fcos * hcos / ((j + 1 - y0) - ftan)
	float d0 = 1;
	m_nFootBottom = fcos * hcos / (d0 - htan) + ftan + y0 - 1;

	return true;
}


bool CameraCoords::InitMiddle()
{
	m_anMiddle = (int*)malloc(m_nImgHeight * sizeof(int));
	if (!m_anMiddle) {
		return false;
	}
	memset(m_anMiddle, 0x00, m_nImgHeight * sizeof(int));

	//float fUnit = ((float)(m_nImgWidth / 2) - (float)m_xVanishing) / ((float)m_nImgHeight - (float)m_yVanishing);
	float fUnit = ((float)(m_nImgWidth / 2) - (float)m_xVanishing) / ((float)m_nFootBottom - (float)m_yVanishing);
	float fMiddle = (float)m_xVanishing + (m_nTopValid - m_yVanishing) * fUnit;
	for (unsigned int i = /*m_yVanishing*/m_nTopValid; i<m_nImgHeight; i++, fMiddle += fUnit) {
		// 车道中线
		m_anMiddle[i] = (int)(fMiddle + 0.5f);
	}

	return true;
}



CameraCoords::CameraCoords(int imgHeight, int imgWidth, int xVanishing, int yVanishing, int nFocus_Dx, int nH)
{
	// 参数错误
	if (imgHeight * imgWidth * nFocus_Dx * nH == 0) {
		m_nImgHeight = 0;
		m_nImgWidth = 0;
		return;
	}

	m_nImgHeight = imgHeight;
	m_nImgWidth = imgWidth;
	m_nFocus = nFocus_Dx;
	m_nH = nH;
	m_xVanishing = xVanishing;
	m_yVanishing = yVanishing;
	m_fTheta = atan((m_yVanishing - 0.5f * m_nImgHeight) / m_nFocus);
	m_nTopValid = (m_yVanishing >= 0) ? m_yVanishing : 0;

	// 计算
	InitCoords();

	InitMiddle();
}



CameraCoords::~CameraCoords()
{
	if (m_anDistance) {
		free(m_anDistance);
	}

	if (m_afWidth) {
		free(m_afWidth);
	}

	if (m_anMiddle) {
		free(m_anMiddle);
	}
}



int CameraCoords::GetDistance(int bottom)
{
	if (bottom <= m_nTopValid || bottom >= m_nImgHeight) {
		return -1;
	}
	else {
		return m_anDistance[bottom];
	}
}



bool CameraCoords::GetDistanceAbeam(int x, int width, int bottom, bool isCenterAlig, int& d)
{
	if (bottom <= m_nTopValid || bottom >= m_nImgHeight) {
		return false;
	}
	else {
		int middle = this->GetMiddle(bottom);
		if (isCenterAlig) {
			// 中心距离
			d = this->GetWidth(bottom, x + width / 2 - middle);
		}
		else {
			// 侧面距离
			if (x > middle)
			{// 右侧
				d = this->GetWidth(bottom, x - middle);
			}
			else if (x + width - 1 < middle)
			{// 左侧
				d = this->GetWidth(bottom, x + width - 1 - middle);
			}
			else
			{// 压中线，距离0
				d = 0;
			}
		}

		return true;
	}
}



int CameraCoords::GetWidth(int bottom, int width)
{
	return (int)(GetWidth_f(bottom, width) + 0.5f);
}

float CameraCoords::GetWidth_f(int bottom, int width)
{
	if (bottom <= m_nTopValid || bottom >= m_nImgHeight) {
		return -2.f;
	}
	else {
		return m_afWidth[bottom] * width;
	}
}



int CameraCoords::GetHeight(int bottom, int height)
{
	return GetWidth(bottom, height);
}

float CameraCoords::GetHeight_f(int bottom, int height)
{
	return GetWidth_f(bottom, height);
}



int CameraCoords::GetMiddle(int bottom)
{
	if (bottom <= m_nTopValid || bottom >= m_nImgHeight) {
		return -1;
	}
	else {
		return m_anMiddle[bottom];
	}
}



int CameraCoords::FindPosition(int nDistance)
{
	int nTop = m_nTopValid + 1;
	int nBottom = m_nImgHeight - 1;

	if ( nDistance >= GetDistance(nTop) ) {
		// 大于最远距离
		return nTop;
	}
	else if ( nDistance <= GetDistance(nBottom) ) {
		// 小于最近距离
		return nBottom;
	}

	// 在系统范围内，采用二分法遍历
	while ( nBottom - nTop > 1 ) {
		int nMiddle = (nBottom + nTop) >> 1;
		int nDistanceMiddle = GetDistance(nMiddle);
		if ( nDistanceMiddle == nDistance ) {
			return nMiddle;
		}
		else if ( nDistanceMiddle < nDistance ) {
			nBottom = nMiddle;
		}
		else {	// nDistanceMiddle > nDistance
			nTop = nMiddle;
		}
	}

	return nBottom;
}