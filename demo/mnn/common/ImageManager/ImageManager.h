#ifndef _IMAGEMANAGER_H_
#define _IMAGEMANAGER_H_

//////////////////////////////////////////////////////////////////////////////////
//	介绍																		//
//		该类针对一图多用的需求开发，适用于图像输入、输出、格式转化的统一管理，	//
//	以减少重复存储和运算。														//
//																				//
//	特点：																		//
//		1. 仅支持同样尺寸的图像保存；											//
//		2. 图像保存类型支持BGR图和Gray图；										//
//		3. 支持多种码流输入，包括YUV和BGR，详见IMAGEMANAGER_FORMAT_XXXX；		//
//		4. 支持设置图像输入的帧率；												//
//		5. 支持获取标记，防止重复调用。											//
//																				//
//  版本号：Version 1.0.0.20220106												//
//////////////////////////////////////////////////////////////////////////////////

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <BSJ_AI_config.h>
#include <StatisticalQueue/StatisticalQueue.h>

#define IMAGEMANAGER_KEEP_TYPE_BGR			(1)
#define IMAGEMANAGER_KEEP_TYPE_GRAY1		(2)
#define IMAGEMANAGER_KEEP_TYPE_GRAY3		(3)
#define IMAGEMANAGER_KEEP_TYPE_GRAY_BGR		(4)
#define IMAGEMANAGER_KEEP_TYPE_YUYV			(5)
#define IMAGEMANAGER_KEEP_TYPE_UYVY			(6)
#define IMAGEMANAGER_KEEP_TYPE_NV21			(7)
#define IMAGEMANAGER_KEEP_TYPE_I420			(8)
#define IMAGEMANAGER_KEEP_TYPE_NV12         (9)

#define IMAGEMANAGER_FORMAT_YUYV			(1)
#define IMAGEMANAGER_FORMAT_UYVY			(2)
#define IMAGEMANAGER_FORMAT_NV21			(3)
#define IMAGEMANAGER_FORMAT_I420			(4)
#define IMAGEMANAGER_FORMAT_RGB565			(5)
#define IMAGEMANAGER_FORMAT_BGR565			(6)
#define IMAGEMANAGER_FORMAT_RGB888			(7)
#define IMAGEMANAGER_FORMAT_BGR888			(8)
#define IMAGEMANAGER_FORMAT_NV12			(9)
#define IMAGEMANAGER_FORMAT_NV16			(10)

#define IMAGEMANAGER_FLAG_SUCCESSFUL			(0)
#define IMAGEMANAGER_FLAG_BAD_PARAMETER			(-1)
#define IMAGEMANAGER_FLAG_FETAL					(-2)
#define IMAGEMANAGER_FLAG_NULL_POINTER			(-3)
#define IMAGEMANAGER_FLAG_NO_NECESSARY			(-4)
#define IMAGEMANAGER_FLAG_FULL_QUEUE			(-5)
#define IMAGEMANAGER_FLAG_BUSY					(-18)

#define IMAGEMANAGER_KEEP_MIN_VOLUME		(3) // 图像管理队列最小长度






class ImageManager {

public:

	//////////////////////////////////////////////////////////////////////////
	//	ImageManager(int, int, int, int, unsigned int)						//
	//	功能:																//
	//		构造函数。														//
	//	输入:																//
	//		nImgWidth		图像宽											//
	//		nImgHeight		图像高											//
	//		nImgKeepType	存储类型，IMAGEMANAGER_KEEP_TYPE_XXX			//
	//		nImgKeepNumber	保存数量，最小值 IMAGEMANAGER_KEEP_MIN_VOLUME	//
	//		nFPS			图像保存间隔，单位：帧/s						//
	//	输出: 																//
	//		无。															//
	//////////////////////////////////////////////////////////////////////////

	ImageManager(int nImgWidthInput, int nImgHeightInput, int nImgKeepType, int nImgKeepNumber, unsigned int nFPS, int nImgWidthKeep = 0, int nImgHeightKeep = 0);

	~ImageManager();


	//////////////////////////////////////////////////////////////////////////
	//	int InputImage(int, char*, int, int, int)							//
	//	功能:																//
	//		输入图像。														//
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

	int InputImage(int dataLength, char* data, int imgHeight, int imgWidth, int format);


	//////////////////////////////////////////////////
	//	bool InputBGR(const Mat&)					//
	//	功能:										//
	//		输入BGR图像。							//
	//	返回值:										//
	//		执行结果。								//
	//	输入:										//
	//		matBGR			BGR图像					//
	//	输出: 										//
	//		无。									//
	//////////////////////////////////////////////////

	bool InputBGR(const cv::Mat& matBGR);


	//////////////////////////////////////////////////////////////
	//	bool GetImage(const Mat&, int)							//
	//	功能:													//
	//		获取图像。											//
	//	返回值:													//
	//		执行结果。											//
	//	输入:													//
	//		nGetID			调用标识。							//
	//						不同任务不同标识，防止重复使用。	//
	//						取负数表示无限制取图。				//
	//	输出: 													//
	//		matImage		图像								//
	//////////////////////////////////////////////////////////////

	bool GetImage(cv::Mat& matImage, int nGetID);


	//////////////////////////////////////////////////
	//	void Reset()								//
	//	功能:										//
	//		重置。									//
	//	返回值:										//
	//		空。									//
	//	输入:										//
	//		无。									//
	//	输出: 										//
	//		无。									//
	//////////////////////////////////////////////////

	void Reset();

	void SaveImage(const std::string& strPath);


protected:

	void Free();

	int GetNewNumber();

	// 验证保存条件：定时 & 实时性
	bool KeepNecessary();


	int		m_nImgWidthInput;
	int		m_nImgHeightInput;
	int		m_nImgWidthKeep;
	int		m_nImgHeightKeep;
	int		m_nImgKeepType;		// 保存类型
	int		m_nImgKeepNumber;	// 保存图片上限
	bool	m_bImgInputLock;	// 插入锁

	SQ_GF::StatisticalQueue<int>* m_hQueueFPS;	// 帧率控制队列
	unsigned int	m_nFPS;
	//uint64_t		m_nInputInterval;	// 定时保存间隔，单位同cv::getTickCount()
	//uint64_t		m_nImgKeepTick;		// 图像保存时刻，单位同cv::getTickCount()
	uint64_t		m_nRepeatFlag;		// 图像重复引用标识
	
	std::vector<cv::Mat>		m_vecImages;	// 图像队列
	int				m_nImgPreNumber;	// 上一次图像保存位置
};

#endif