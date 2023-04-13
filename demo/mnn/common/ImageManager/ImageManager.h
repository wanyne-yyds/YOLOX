#ifndef _IMAGEMANAGER_H_
#define _IMAGEMANAGER_H_

//////////////////////////////////////////////////////////////////////////////////
//	����																		//
//		�������һͼ���õ����󿪷���������ͼ�����롢�������ʽת����ͳһ����	//
//	�Լ����ظ��洢�����㡣														//
//																				//
//	�ص㣺																		//
//		1. ��֧��ͬ���ߴ��ͼ�񱣴棻											//
//		2. ͼ�񱣴�����֧��BGRͼ��Grayͼ��										//
//		3. ֧�ֶ����������룬����YUV��BGR�����IMAGEMANAGER_FORMAT_XXXX��		//
//		4. ֧������ͼ�������֡�ʣ�												//
//		5. ֧�ֻ�ȡ��ǣ���ֹ�ظ����á�											//
//																				//
//  �汾�ţ�Version 1.0.0.20220106												//
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

#define IMAGEMANAGER_KEEP_MIN_VOLUME		(3) // ͼ����������С����






class ImageManager {

public:

	//////////////////////////////////////////////////////////////////////////
	//	ImageManager(int, int, int, int, unsigned int)						//
	//	����:																//
	//		���캯����														//
	//	����:																//
	//		nImgWidth		ͼ���											//
	//		nImgHeight		ͼ���											//
	//		nImgKeepType	�洢���ͣ�IMAGEMANAGER_KEEP_TYPE_XXX			//
	//		nImgKeepNumber	������������Сֵ IMAGEMANAGER_KEEP_MIN_VOLUME	//
	//		nFPS			ͼ�񱣴�������λ��֡/s						//
	//	���: 																//
	//		�ޡ�															//
	//////////////////////////////////////////////////////////////////////////

	ImageManager(int nImgWidthInput, int nImgHeightInput, int nImgKeepType, int nImgKeepNumber, unsigned int nFPS, int nImgWidthKeep = 0, int nImgHeightKeep = 0);

	~ImageManager();


	//////////////////////////////////////////////////////////////////////////
	//	int InputImage(int, char*, int, int, int)							//
	//	����:																//
	//		����ͼ��														//
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

	int InputImage(int dataLength, char* data, int imgHeight, int imgWidth, int format);


	//////////////////////////////////////////////////
	//	bool InputBGR(const Mat&)					//
	//	����:										//
	//		����BGRͼ��							//
	//	����ֵ:										//
	//		ִ�н����								//
	//	����:										//
	//		matBGR			BGRͼ��					//
	//	���: 										//
	//		�ޡ�									//
	//////////////////////////////////////////////////

	bool InputBGR(const cv::Mat& matBGR);


	//////////////////////////////////////////////////////////////
	//	bool GetImage(const Mat&, int)							//
	//	����:													//
	//		��ȡͼ��											//
	//	����ֵ:													//
	//		ִ�н����											//
	//	����:													//
	//		nGetID			���ñ�ʶ��							//
	//						��ͬ����ͬ��ʶ����ֹ�ظ�ʹ�á�	//
	//						ȡ������ʾ������ȡͼ��				//
	//	���: 													//
	//		matImage		ͼ��								//
	//////////////////////////////////////////////////////////////

	bool GetImage(cv::Mat& matImage, int nGetID);


	//////////////////////////////////////////////////
	//	void Reset()								//
	//	����:										//
	//		���á�									//
	//	����ֵ:										//
	//		�ա�									//
	//	����:										//
	//		�ޡ�									//
	//	���: 										//
	//		�ޡ�									//
	//////////////////////////////////////////////////

	void Reset();

	void SaveImage(const std::string& strPath);


protected:

	void Free();

	int GetNewNumber();

	// ��֤������������ʱ & ʵʱ��
	bool KeepNecessary();


	int		m_nImgWidthInput;
	int		m_nImgHeightInput;
	int		m_nImgWidthKeep;
	int		m_nImgHeightKeep;
	int		m_nImgKeepType;		// ��������
	int		m_nImgKeepNumber;	// ����ͼƬ����
	bool	m_bImgInputLock;	// ������

	SQ_GF::StatisticalQueue<int>* m_hQueueFPS;	// ֡�ʿ��ƶ���
	unsigned int	m_nFPS;
	//uint64_t		m_nInputInterval;	// ��ʱ����������λͬcv::getTickCount()
	//uint64_t		m_nImgKeepTick;		// ͼ�񱣴�ʱ�̣���λͬcv::getTickCount()
	uint64_t		m_nRepeatFlag;		// ͼ���ظ����ñ�ʶ
	
	std::vector<cv::Mat>		m_vecImages;	// ͼ�����
	int				m_nImgPreNumber;	// ��һ��ͼ�񱣴�λ��
};

#endif