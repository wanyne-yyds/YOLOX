#include "ImageManagerRga.h"

ImageManagerRga::ImageManagerRga(int nImgWidthInput, int nImgHeightInput, int nImgKeepType, int nImgKeepNumber, unsigned int nFPS, int nImgWidthKeep, int nImgHeightKeep)
 : ImageManager(nImgWidthInput, nImgHeightInput, nImgKeepType, nImgKeepNumber, nFPS, nImgWidthKeep, nImgHeightKeep)
{
	//RV1126_RGA
	//memset(&m_rga_src, 0, sizeof(m_rga_src));
	//memset(&m_rga_dst, 0, sizeof(m_rga_dst));
}


ImageManagerRga::~ImageManagerRga()
{
}


RgaSURF_FORMAT ImageManagerRga::rgaFormat(int format)
{
	switch (format) {
	case IMAGEMANAGER_FORMAT_YUYV:
		return RK_FORMAT_YUYV_422;
	case IMAGEMANAGER_FORMAT_UYVY:
		return RK_FORMAT_UYVY_422;
	case IMAGEMANAGER_FORMAT_NV21:
		return RK_FORMAT_YCrCb_420_SP;
	case IMAGEMANAGER_FORMAT_I420:
		return RK_FORMAT_YCrCb_420_P;
	case IMAGEMANAGER_FORMAT_RGB565:
		return RK_FORMAT_RGB_565;
	case IMAGEMANAGER_FORMAT_BGR565:
		return RK_FORMAT_RGB_565;
	case IMAGEMANAGER_FORMAT_RGB888:
		return RK_FORMAT_RGB_888;
	case IMAGEMANAGER_FORMAT_BGR888:
		return RK_FORMAT_BGR_888;
    case IMAGEMANAGER_FORMAT_NV12:
		return RK_FORMAT_YCbCr_420_SP;
	case IMAGEMANAGER_FORMAT_NV16:
		return RK_FORMAT_YCbCr_422_SP;
	default:
		LOGE("ImageManagerRga::InputImageRga error format: %d.\n", format);
		return RK_FORMAT_UNKNOWN;
	}
}



//RV1126_RGA
int ImageManagerRga::InputImageRga(int dataLength, char* data, int imgHeight, int imgWidth, int format)
{
	RgaSURF_FORMAT nSrcFormatRga = this->rgaFormat(format);
	if (nSrcFormatRga == RK_FORMAT_UNKNOWN) {
		LOGE("ImageManagerRga::InputImageRga err: bad format = %d!\n", format);
		return IMAGEMANAGER_FLAG_BAD_PARAMETER;
	}
	
	// 参数判断
	if (data == NULL) {
		BSJ_AI::sleep(10);
		LOGI("ImageManagerRga::InputImageRga err: data is NULL!\n");
		return IMAGEMANAGER_FLAG_NULL_POINTER;
	}
	else if (imgHeight != m_nImgHeightInput || imgWidth != m_nImgWidthInput) {
		BSJ_AI::sleep(10);
		LOGE("ImageManagerRga::InputImageRga err: bad size(%d,%d)!\n", imgWidth, imgHeight);
		return IMAGEMANAGER_FLAG_BAD_PARAMETER;
	}


	// 限定帧率
	if (!this->KeepNecessary()) {
		BSJ_AI::sleep(10);
		LOGI("ImageManagerRga::InputImageRga err: InputImage no necessary!\n");
		return IMAGEMANAGER_FLAG_NO_NECESSARY;
	}

	// 判断数据长度
	switch (format) {
	case IMAGEMANAGER_FORMAT_YUYV:
	case IMAGEMANAGER_FORMAT_UYVY:
	case IMAGEMANAGER_FORMAT_RGB565:
	case IMAGEMANAGER_FORMAT_BGR565:
	case IMAGEMANAGER_FORMAT_NV16:
		if (imgHeight * imgWidth * 2 != dataLength) {
			BSJ_AI::sleep(10);
			LOGE("ImageManagerRga::InputImageRga err: bad length of data, format=%d, height * width * 2 != dataLength, dataLength=%d, size(%d,%d)", format, dataLength, imgWidth, imgHeight);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
		break;
	case IMAGEMANAGER_FORMAT_NV21:
	case IMAGEMANAGER_FORMAT_NV12:
	case IMAGEMANAGER_FORMAT_I420:
		if (imgHeight * imgWidth * 3 / 2 != dataLength) {
			BSJ_AI::sleep(10);
			LOGE("ImageManagerRga::InputImageRga err: bad length of data, format=%d, height * width * 3 / 2 != dataLength, dataLength=%d, size(%d,%d)", format, dataLength, imgWidth, imgHeight);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
		break;
	case IMAGEMANAGER_FORMAT_RGB888:
	case IMAGEMANAGER_FORMAT_BGR888:
		if (imgHeight * imgWidth * 3 != dataLength) {
			BSJ_AI::sleep(10);
			LOGE("ImageManagerRga::InputImageRga err: bad length of data, format=%d, height * width * 2 != dataLength, dataLength=%d, size(%d,%d)", format, dataLength, imgWidth, imgHeight);
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
		break;
	default:
		BSJ_AI::sleep(10);
		LOGE("ImageManagerRga::InputImageRga err: format: %d.\n", format);
		return IMAGEMANAGER_FLAG_BAD_PARAMETER;
	}
	

	// 新序号
	int number = this->GetNewNumber();
	if (number < 0) {
		BSJ_AI::sleep(10);
		LOGE("ImageManagerRga::InputImageRga err: GetNewNumber failed!");
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

	rga_buffer_t rga_src = wrapbuffer_virtualaddr(data, imgWidth, imgHeight, nSrcFormatRga);
	rga_src.format = nSrcFormatRga;

	cv::Mat matData;
	if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_BGR)
	{
		//m_vecImages[number] = cv::Mat(imgHeight, imgWidth, CV_8UC3);
		matData = cv::Mat(imgHeight, imgWidth, CV_8UC3);
		// rga_buffer_t rga_dst = wrapbuffer_virtualaddr(m_vecImages[number].data, imgWidth, imgHeight, RK_FORMAT_BGR_888);
		rga_buffer_t rga_dst = wrapbuffer_virtualaddr(matData.data, imgWidth, imgHeight, RK_FORMAT_BGR_888);
		rga_dst.format = RK_FORMAT_BGR_888;
		IM_STATUS status = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
		if(status != IM_STATUS_SUCCESS)	{
			LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_FETAL;
		}
		// resize
		if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
			//cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidth, m_nImgHeight));
			rga_buffer_t src = wrapbuffer_virtualaddr(matData.data, matData.cols, matData.rows, RK_FORMAT_BGR_888);
			rga_buffer_t dst = wrapbuffer_virtualaddr(m_vecImages[number].data, m_nImgWidthKeep, m_nImgHeightKeep, RK_FORMAT_BGR_888);

			IM_STATUS status = imresize(src, dst);
			if (status != IM_STATUS_SUCCESS){
				LOGE("ImageManagerRga::InputImageRga err: rga imresize status = %d.\n", status);
				m_bImgInputLock = false;
				return IMAGEMANAGER_FLAG_FETAL;
			}
		}
		else {
			m_vecImages[number] = matData;
		}
		
	}
	else if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_GRAY1)
	{
		matData = cv::Mat(imgHeight, imgWidth, CV_8UC1);
		rga_buffer_t rga_dst = wrapbuffer_virtualaddr(matData.data, imgWidth, imgHeight, RK_FORMAT_YCbCr_400);
		rga_dst.format = RK_FORMAT_YCbCr_400;
		IM_STATUS status = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
		if(status != IM_STATUS_SUCCESS)	{
			LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_FETAL;
		}

		// resize
		if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
			//cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidth, m_nImgHeight));
			rga_buffer_t src = wrapbuffer_virtualaddr(matData.data, matData.cols, matData.rows, RK_FORMAT_YCbCr_400);
			rga_buffer_t dst = wrapbuffer_virtualaddr(m_vecImages[number].data, m_nImgWidthKeep, m_nImgHeightKeep, RK_FORMAT_YCbCr_400);

			IM_STATUS status = imresize(src, dst);
			if (status != IM_STATUS_SUCCESS){
				LOGE("ImageManagerRga::InputImageRga err: rga imresize status = %d.\n", status);
				m_bImgInputLock = false;
				return IMAGEMANAGER_FLAG_FETAL;
			}
		}
		else {
			m_vecImages[number] = matData;
		}
	}
	else if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_GRAY3)
	{
		cv::Mat matGray = cv::Mat(imgHeight, imgWidth, CV_8UC1);
		rga_buffer_t rga_dst = wrapbuffer_virtualaddr(matGray.data, imgWidth, imgHeight, RK_FORMAT_YCbCr_400);
		rga_dst.format = RK_FORMAT_YCbCr_400;
		IM_STATUS status = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
		// if (status == IM_STATUS_SUCCESS) {
		// 	cvtColor(matGray, m_vecImages[number], CV_GRAY2BGR);
    	// }
		// else {
		// 	LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
		// 	m_bImgInputLock = false;
		// 	return IMAGEMANAGER_FLAG_FETAL;
		// }

		if (status != IM_STATUS_SUCCESS) {
			LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_FETAL;
		}
		
		// resize
		if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
			//cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidth, m_nImgHeight));
			matData = cv::Mat(m_nImgHeightKeep, m_nImgWidthKeep, CV_8UC1);
			rga_buffer_t src = wrapbuffer_virtualaddr(matGray.data, matGray.cols, matGray.rows, RK_FORMAT_YCbCr_400);
			rga_buffer_t dst = wrapbuffer_virtualaddr(matData.data, m_nImgWidthKeep, m_nImgHeightKeep, RK_FORMAT_YCbCr_400);

			IM_STATUS status = imresize(src, dst);
			if (status != IM_STATUS_SUCCESS){
				LOGE("ImageManagerRga::InputImageRga err: rga imresize status = %d.\n", status);
				m_bImgInputLock = false;
				return IMAGEMANAGER_FLAG_FETAL;
			}
			matGray = matData;
		}
		
		cvtColor(matGray, m_vecImages[number], CV_GRAY2BGR);

		matGray.release();
	}
	else if (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_GRAY_BGR)
	{
		int w = BSJ_AI::getRounding(imgWidth, 12, BSJ_AI::ROUNDING_TYPE::ROUND);
		int h = imgHeight / 3;
		cv::Mat matResize;
		switch (format) {
		case IMAGEMANAGER_FORMAT_YUYV:
		case IMAGEMANAGER_FORMAT_UYVY:
		case IMAGEMANAGER_FORMAT_NV16:
		case IMAGEMANAGER_FORMAT_RGB565:
		case IMAGEMANAGER_FORMAT_BGR565:
			matResize = cv::Mat(h, w, CV_8UC2);
			break;
		case IMAGEMANAGER_FORMAT_NV21:
		case IMAGEMANAGER_FORMAT_I420:
	    case IMAGEMANAGER_FORMAT_NV12:
			matResize = cv::Mat(h + h / 2, w, CV_8UC1);
			break;
		case IMAGEMANAGER_FORMAT_RGB888:
		case IMAGEMANAGER_FORMAT_BGR888:
			matResize = cv::Mat(h, w, CV_8UC3);
			break;
		default:
			LOGE("ImageManagerRga::InputImageRga err: format: %d.\n", format);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}

		rga_buffer_t rga_resize = wrapbuffer_virtualaddr(matResize.data, w, h, nSrcFormatRga);
		rga_resize.format = nSrcFormatRga;
		IM_STATUS status = imresize(rga_src, rga_resize);
		if (status != IM_STATUS_SUCCESS) {
			LOGE("ImageManagerRga::InputImageRga err: rga imresize status = %d.\n", status);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_FETAL;
		}

		matData = cv::Mat(h, w / 3, CV_8UC3);
		rga_buffer_t rga_dst = wrapbuffer_virtualaddr(matData.data, w, h, RK_FORMAT_YCbCr_400);
		rga_dst.format = RK_FORMAT_YCbCr_400;
		status = imcvtcolor(rga_resize, rga_dst, rga_resize.format, rga_dst.format);
		if (status != IM_STATUS_SUCCESS) {
			LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_FETAL;
		}

		// resize
		if (imgHeight != m_nImgHeightKeep || imgWidth != m_nImgWidthKeep) {
			//cv::resize(m_vecImages[number], m_vecImages[number], cv::Size(m_nImgWidth, m_nImgHeight));
			rga_buffer_t src = wrapbuffer_virtualaddr(matData.data, matData.cols, matData.rows, RK_FORMAT_YCbCr_400);
			rga_buffer_t dst = wrapbuffer_virtualaddr(m_vecImages[number].data, m_nImgWidthKeep, m_nImgHeightKeep, RK_FORMAT_YCbCr_400);

			IM_STATUS status = imresize(src, dst);
			if (status != IM_STATUS_SUCCESS){
				LOGE("ImageManagerRga::InputImageRga err: rga imresize status = %d.\n", status);
				m_bImgInputLock = false;
				return IMAGEMANAGER_FLAG_FETAL;
			}
		}
		else {
			m_vecImages[number] = matData;
		}
	}
	else {
		if ((m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_YUYV && format == IMAGEMANAGER_FORMAT_YUYV)
			|| (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_UYVY && format == IMAGEMANAGER_FORMAT_UYVY))
		{
			m_vecImages[number] = cv::Mat(imgHeight, imgWidth, CV_8UC2);
			rga_buffer_t rga_dst = wrapbuffer_virtualaddr(m_vecImages[number].data, m_nImgWidthKeep, m_nImgHeightKeep, nSrcFormatRga);
			rga_dst.format = nSrcFormatRga;
			IM_STATUS status = imresize(rga_src, rga_dst);
			if (status != IM_STATUS_SUCCESS) {
				LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
				m_bImgInputLock = false;
				return IMAGEMANAGER_FLAG_FETAL;
			}
		}
		else if ((m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_NV21 && format == IMAGEMANAGER_FORMAT_NV21)
			|| (m_nImgKeepType == IMAGEMANAGER_KEEP_TYPE_I420 && format == IMAGEMANAGER_FORMAT_I420)) {
			m_vecImages[number] = cv::Mat(imgHeight + (imgHeight >> 1), imgWidth, CV_8UC1);
			rga_buffer_t rga_dst = wrapbuffer_virtualaddr(m_vecImages[number].data, m_nImgWidthKeep, m_nImgHeightKeep, nSrcFormatRga);
			rga_dst.format = nSrcFormatRga;
			IM_STATUS status = imresize(rga_src, rga_dst);
			if (status != IM_STATUS_SUCCESS) {
				LOGE("ImageManagerRga::InputImageRga err: rga imcvtcolor status = %d.\n", status);
				m_bImgInputLock = false;
				return IMAGEMANAGER_FLAG_FETAL;
			}
		}
		else {
			BSJ_AI::sleep(1000);
			LOGE("ImageManagerRga::InputImageRga err: bad format(%d).\n", format);
			m_bImgInputLock = false;
			return IMAGEMANAGER_FLAG_BAD_PARAMETER;
		}
	}
	matData.release();
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

	return IMAGEMANAGER_FLAG_SUCCESSFUL;
}

