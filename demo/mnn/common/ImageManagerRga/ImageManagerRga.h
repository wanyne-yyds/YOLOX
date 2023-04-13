#ifndef _IMAGEMANAGERRGA_H_
#define _IMAGEMANAGERRGA_H_

#include "../ImageManager/ImageManager.h"

//RV1126_RGA 
#include "im2d_api/im2d.hpp"
#include "RgaUtils.h"
#include "rga.h"

class ImageManagerRga : public ImageManager{
public:
	//////////////////////////////////////////////////////////////////////////
	//	ImageManagerRga(int, int, int, int, int)								//
	//	����:																//
	//		Rga���캯����														//
	//	����:																//
	//		nImgWidth		ͼ���											//
	//		nImgHeight		ͼ���											//
	//		nImgKeepType	�洢���ͣ�IMAGEMANAGER_KEEP_TYPE_XXX			//
	//		nImgKeepNumber	������������Сֵ IMAGEMANAGER_KEEP_MIN_VOLUME	//
	//		nFPS			ͼ�񱣴�������λ��֡/s						//
	//	���: 																//
	//		�ޡ�															//
	//////////////////////////////////////////////////////////////////////////

	ImageManagerRga(int nImgWidthInput, int nImgHeightInput, int nImgKeepType, int nImgKeepNumber, unsigned int nFPS, int nImgWidthKeep = 0, int nImgHeightKeep = 0);
	~ImageManagerRga();
	
	//////////////////////////////////////////////////////////////////////////
	//	int InputImageRga(int, char*, int, int, int)						//
	//	����:																//
	//		����ͼ��,��ʹ��rgaת���� 											//
	//	����ֵ:																//
	//		ִ�н����														//
	//	����:																//
	//		dataLength		���ݳ���										//
	//		data			����											//
	//		imgHeight		ͼ���											//
	//		imgWidth		ͼ���											//
	//		format			�������ͣ�IMAGEMANAGER_FORMAT_XXX				//
	//	���: 																//
	//		�ޡ�															//
	//////////////////////////////////////////////////////////////////////////
	int InputImageRga(int dataLength, char* data, int imgHeight, int imgWidth, int format);
	
private:

	RgaSURF_FORMAT rgaFormat(int format);
	
	//RV1126_RGA
	//rga_buffer_t m_rga_src;
	//rga_buffer_t m_rga_dst;
};

#endif
