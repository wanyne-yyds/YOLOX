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
	//	功能:																//
	//		Rga构造函数。														//
	//	输入:																//
	//		nImgWidth		图像宽											//
	//		nImgHeight		图像高											//
	//		nImgKeepType	存储类型，IMAGEMANAGER_KEEP_TYPE_XXX			//
	//		nImgKeepNumber	保存数量，最小值 IMAGEMANAGER_KEEP_MIN_VOLUME	//
	//		nFPS			图像保存间隔，单位：帧/s						//
	//	输出: 																//
	//		无。															//
	//////////////////////////////////////////////////////////////////////////

	ImageManagerRga(int nImgWidthInput, int nImgHeightInput, int nImgKeepType, int nImgKeepNumber, unsigned int nFPS, int nImgWidthKeep = 0, int nImgHeightKeep = 0);
	~ImageManagerRga();
	
	//////////////////////////////////////////////////////////////////////////
	//	int InputImageRga(int, char*, int, int, int)						//
	//	功能:																//
	//		输入图像,并使用rga转换。 											//
	//	返回值:																//
	//		执行结果。														//
	//	输入:																//
	//		dataLength		数据长度										//
	//		data			数据											//
	//		imgHeight		图像高											//
	//		imgWidth		图像宽											//
	//		format			数据类型，IMAGEMANAGER_FORMAT_XXX				//
	//	输出: 																//
	//		无。															//
	//////////////////////////////////////////////////////////////////////////
	int InputImageRga(int dataLength, char* data, int imgHeight, int imgWidth, int format);
	
private:

	RgaSURF_FORMAT rgaFormat(int format);
	
	//RV1126_RGA
	//rga_buffer_t m_rga_src;
	//rga_buffer_t m_rga_dst;
};

#endif
