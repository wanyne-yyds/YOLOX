#include "rknn.h"
#include "im2d_api/im2d.hpp"
#include "RgaUtils.h"
#include "rga.h"

BSJ_AI::RKNN::RKNN()
{
	m_hContext = 0;
	m_nIoNumber.n_input = 0;
	m_nIoNumber.n_output = 0;
	m_aInputAttr = 0;
	m_aInputData = 0;
	m_aOutputAttr = 0;
	m_aOutputData = 0;

	m_bRunningLock = false;
	m_bInitLock = false;
}


BSJ_AI::RKNN::~RKNN()
{
	this->free();
}


int BSJ_AI::RKNN::free()
{
	// wait for running
	while (m_bRunningLock) {
		BSJ_AI::sleep(100);
	}
	
	if (m_hContext) {
		int ret = rknn_destroy(m_hContext);
		m_hContext = 0;
	}


	if (m_aInputAttr) {
		delete m_aInputAttr;
		m_aInputAttr = 0;
	}
	if (m_aInputData) {
		for (int i = m_nIoNumber.n_input - 1; i >= 0; i--) {
			delete ((uint8_t*)m_aInputData[i].buf);
		}
		delete m_aInputData;
		m_aInputData = 0;
	}
	if (m_aOutputAttr) {
		delete m_aOutputAttr;
		m_aOutputAttr = 0;
	}
	if (m_aOutputData) {
		for (int i = m_nIoNumber.n_output - 1; i >= 0; i--) {
			delete ((uint8_t*)m_aOutputData[i].buf);
		}
		delete m_aOutputData;
		m_aOutputData = 0;
	}
	
	m_nIoNumber.n_input = 0;
	m_nIoNumber.n_output = 0;


	m_bRunningLock = false;
	m_bInitLock = false;

	return 0;
}


BSJ_AI::NCHW getNCHW(const rknn_tensor_attr& tensor)
{
	switch (tensor.fmt) {
	case RKNN_TENSOR_NCHW:
		return BSJ_AI::NCHW(tensor.dims[3], tensor.dims[2], tensor.dims[1], tensor.dims[0]);
	case RKNN_TENSOR_NHWC:
	default:
		return BSJ_AI::NCHW(tensor.dims[3], tensor.dims[1], tensor.dims[0], tensor.dims[2]);
	}
}

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

int BSJ_AI::RKNN::init(int modelSize, unsigned char* model, uint32_t flag)
{
	// lock
	if (m_bInitLock) {
		LOGE("BSJ_AI::rknn::init err: Initialization in progress! Don't repeat!\n");
		return RKNN_ERR_FAIL;
	}
	else {
		m_bInitLock = true;
	}
	this->free();

	// rknn_init
	int ret = rknn_init(&m_hContext, model, modelSize, flag);
	if (ret != RKNN_SUCC) {
		LOGE("rknn::init err: rknn_init failed with %d.\n", ret);
		// unlock
		m_bInitLock = false;
		return ret;
	}
	
	// rknn_query io num
	ret = rknn_query(m_hContext, RKNN_QUERY_IN_OUT_NUM, &m_nIoNumber, sizeof(m_nIoNumber));
	// printf("model input num: %d, output num: %d\n", m_nIoNumber.n_input, m_nIoNumber.n_output);
	if (ret != RKNN_SUCC) {
		LOGE("rknn::init err: rknn_query(RKNN_QUERY_IN_OUT_NUM) failed with %d.\n", ret);
		this->free();
		// unlock
		m_bInitLock = false;
		return ret;
	}
	else if (m_nIoNumber.n_input != 1) {
		// support if and only if single input tensor
		LOGE("BSJ_AI::rknn::run err: the number of input tensor is %d. Support if and only if single input tensor!\n", m_nIoNumber.n_input);
		this->free();
		// unlock
		m_bInitLock = false;
		return RKNN_ERR_MODEL_INVALID;
	}

	// rknn_query input attr & malloc rknn_input
	m_aInputAttr = new rknn_tensor_attr[m_nIoNumber.n_input];
	memset(m_aInputAttr, 0, m_nIoNumber.n_input * sizeof(rknn_tensor_attr));
	m_aInputData = new rknn_input[m_nIoNumber.n_input];
	memset(m_aInputData, 0, m_nIoNumber.n_input * sizeof(rknn_input));
	for (int i = 0; i < m_nIoNumber.n_input; i++) {
		m_aInputAttr[i].index = i;
 		ret = rknn_query(m_hContext, RKNN_QUERY_INPUT_ATTR, &(m_aInputAttr[i]), sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			LOGE("rknn::init err: rknn_query(RKNN_QUERY_INPUT_ATTR, %d) failed with %d.\n", i, ret);
			this->free();
			// unlock
			m_bInitLock = false;
			return ret;
		}
		else {
			BSJ_AI::NCHW nchw = getNCHW(m_aInputAttr[i]);
			
			//printf("dims=[%d %d %d %d], fmt = %d", m_aInputAttr[i].dims[3], m_aInputAttr[i].dims[2], m_aInputAttr[i].dims[1], m_aInputAttr[i].dims[0], m_aInputAttr[i].fmt);
			
			if (nchw.c != 3) {
				LOGE("BSJ_AI::rknn::init err: the channels of model must be 3!\n");
				// unlock
				m_bInitLock = false;
				return RKNN_ERR_MODEL_INVALID;
			}
		}
		// rknn_input
		m_aInputData[i].index = m_aInputAttr[i].index;
		m_aInputData[i].size = m_aInputAttr[i].size;
		m_aInputData[i].buf = (void*)new uint8_t[m_aInputData[i].size];
		m_aInputData[i].type = RKNN_TENSOR_UINT8;
		m_aInputData[i].fmt = RKNN_TENSOR_NHWC;
		m_aInputData[i].pass_through = 0;
		printRKNNTensor(&(m_aInputAttr[i]));
	}
	// rknn_query output attr & malloc rknn_output
	m_aOutputAttr = new rknn_tensor_attr[m_nIoNumber.n_output];
	memset(m_aOutputAttr, 0, m_nIoNumber.n_output * sizeof(rknn_tensor_attr));
	m_aOutputData = new rknn_output[m_nIoNumber.n_output];
	memset(m_aOutputData, 0, m_nIoNumber.n_output * sizeof(rknn_output));
	for (int i = 0; i < m_nIoNumber.n_output; i++) {
		m_aOutputAttr[i].index = i;
 		ret = rknn_query(m_hContext, RKNN_QUERY_OUTPUT_ATTR, &(m_aOutputAttr[i]), sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			LOGE("rknn::init err: rknn_query(RKNN_QUERY_OUTPUT_ATTR, %d) failed with %d.\n", i, ret);
			this->free();
			// unlock
			m_bInitLock = false;
			return ret;
		}
		// rknn_output
		m_aOutputData[i].index = m_aOutputAttr[i].index;
		m_aOutputData[i].want_float = 1;
		m_aOutputData[i].is_prealloc = 1;
		switch (m_aOutputAttr[i].type) {
		case RKNN_TENSOR_FLOAT32:
			m_aOutputData[i].size = m_aOutputAttr[i].size;
			break;
    	case RKNN_TENSOR_FLOAT16:
		case RKNN_TENSOR_INT16:
			m_aOutputData[i].size = m_aOutputAttr[i].size * 2;
			break;
    	case RKNN_TENSOR_INT8:
    	case RKNN_TENSOR_UINT8:
			m_aOutputData[i].size = m_aOutputAttr[i].size * 4;
			break;
		default:
			LOGE("rknn::init err: m_aOutputAttr[%d].type = %d.\n", i, m_aOutputAttr[i].type);
			this->free();
			// unlock
			m_bInitLock = false;
			return RKNN_ERR_FAIL;
		}
		m_aOutputData[i].buf = (void*)new uint8_t[m_aOutputData[i].size];

		//printf("m_aOutputData[i].size = %d\n", m_aOutputData[i].size);
		printRKNNTensor(&(m_aOutputAttr[i]));
	}

	// unlock
	m_bInitLock = false;
	
	return RKNN_SUCC;
}

bool indexIn(int index, const std::vector<int>& vecCoordsIndex)
{
	for (std::vector<int>::const_iterator it = vecCoordsIndex.begin(); it != vecCoordsIndex.end(); it++) {
		if (*it == index) {
			return true;
		}
	}

	return false;
}

bool BSJ_AI::RKNN::cropImage(const cv::Mat& img, const cv::Rect& r, void* dst_buf, const cv::Size& resize_wh)
{
	if (r.width <= 0 || r.height <= 0) {
		return false;
	}
		

	cv::Rect rect_image(0, 0, img.cols, img.rows);
	cv::Mat matA = img;
	if (r != rect_image) {
		cv::Mat matA = img(r).clone();
	}
		
	// 压缩
	rga_buffer_t src = wrapbuffer_virtualaddr(matA.data, matA.cols, matA.rows, RK_FORMAT_BGR_888);
	rga_buffer_t dst = wrapbuffer_virtualaddr(dst_buf, resize_wh.width, resize_wh.height, RK_FORMAT_BGR_888);

	IM_STATUS status = imresize(src, dst);
	if (status != IM_STATUS_SUCCESS){
		LOGE("BSJ_AI::RKNN::cropImage imresize error status = %d......\n", status);
		return false;
	}
	return true;
}

int BSJ_AI::RKNN::run(const std::vector<BSJ_AI::ROI_DATA>& vecBatchData, 
					const std::vector<int>& vecCoordsIndex, 
					std::vector<std::vector<float> >& vecOutputs,
					std::function< void( int, void*, int ) > call_back, 
					bool useRGA)
{
	if (m_bInitLock) {
		BSJ_AI::sleep(1000);
		LOGE("BSJ_AI::rknn::run err: Initialization in progress! Wait for a moment!\n");
		return RKNN_ERR_FAIL;
	}
	else if (m_bRunningLock) {
		BSJ_AI::sleep(1000);
		LOGE("BSJ_AI::rknn::run err: Running in progress! Don't repeat!\n");
		return RKNN_ERR_FAIL;
	}

	
	if (!m_hContext) {
		LOGE("BSJ_AI::rknn::run err: system is not uninitialized!\n");
		return RKNN_ERR_CTX_INVALID;
	}
	else if (m_nIoNumber.n_input != 1) {
		// support if and only if single input tensor
		LOGE("BSJ_AI::rknn::run err: the number of input tensor is %d. Support if and only if single input tensor!\n", m_nIoNumber.n_input);
		this->free();
		return RKNN_ERR_MODEL_INVALID;
	}

	// m_bRunningLock = false;
	
	vecOutputs.clear();
	
	// net params
	BSJ_AI::NCHW nchwInput = getNCHW(m_aInputAttr[0]);
	
	// check data
	int szBatch = BSJ_MIN(vecBatchData.size(), nchwInput.n);
	if (szBatch == 0) {
		LOGE("BSJ_AI::rknn::run err: vecBatchData.size() == 0.\n");
		// unlock
		//m_bRunningLock = false;
		return RKNN_ERR_INPUT_INVALID;
	}
	for (int i = 0; i<szBatch; i++) {
		if (vecBatchData[i].image.channels() != 3) {
			LOGE("BSJ_AI::rknn::run err: the channels of image must be 3!\n");
			// unlock
			//m_bRunningLock = false;
			return RKNN_ERR_INPUT_INVALID;
		}
	}
	
	// Set Input Data
	cv::Mat matResize;
	for (int i = 0; i < szBatch; i++) {
		bool bResult;
		if(useRGA) {
			bResult = this->cropImage(vecBatchData[i].image, vecBatchData[i].roi, m_aInputData[i].buf, cv::Size(nchwInput.w, nchwInput.h));
		}
		else {
			matResize = cv::Mat(nchwInput.h, nchwInput.w, CV_8UC3, m_aInputData[i].buf + (nchwInput.h * nchwInput.w * 3 * i));
			bResult = BSJ_AI::cropImage(vecBatchData[i].image, vecBatchData[i].roi, matResize, cv::Size(nchwInput.w, nchwInput.h));
			//cv::imwrite("matResize.jpg", matResize);
		}
		
		 if (!bResult) {
		 	LOGE("BSJ_AI::rknn::run err: BSJ_AI::cropImage failed! image[%d*%d]\n", vecBatchData[i].image.cols, vecBatchData[i].image.rows);
		 }
	}

	int nResult = rknn_inputs_set(m_hContext, m_nIoNumber.n_input, m_aInputData);
	if(nResult < 0) {
		LOGE("BSJ_AI::rknn::run err: rknn_input_set fail! ret = %d.\n", nResult);
		// unlock
		//m_bRunningLock = false;
		return nResult;
	}

	// Run
	nResult = rknn_run(m_hContext, nullptr);
	if(nResult < 0) {
		LOGE("BSJ_AI::rknn::run err: rknn_run fail! ret = %d.\n", nResult);
		// unlock
		//m_bRunningLock = false;
		return nResult;
	}

	
	// Get Output
	nResult = rknn_outputs_get(m_hContext, m_nIoNumber.n_output, m_aOutputData, NULL);
	if(nResult < 0) {
		LOGE("BSJ_AI::rknn::run err: rknn_outputs_get fail! ret = %d.\n", nResult);
		// unlock
		//m_bRunningLock = false;
		return nResult;
	}


	
	// Post Process
	for (int i = 0; i < m_nIoNumber.n_output; i++) {
		std::vector<float> retTemp;
		int nCount = m_aOutputAttr[i].n_elems;	// m_aOutputData[i].size >> 2

		if  (call_back) {
			call_back(i, m_aOutputData[i].buf, nCount);
			continue;
		}

		for (int j = 0; j < nCount; j++) {
			// if(i==0){
			// 	if(((float*)(m_aOutputData[i].buf))[j] > 0.1)
			// 		LOGE("%.6f\n", ((float*)(m_aOutputData[i].buf))[j]);
			// }
			retTemp.push_back( ((float*)(m_aOutputData[i].buf))[j] );
		}

		// coord recovery
		if (indexIn(i, vecCoordsIndex))
		{
			int nCountPerBatch = retTemp.size() / nchwInput.n;	// 每个batch输出坐标数量

			// 按batch分别还原坐标
			for (int u = 0; u < nchwInput.n; u++)
			{
				const cv::Rect& recRoi = vecBatchData[u].roi;	// 当前batch的ROI
				float wCoeff = recRoi.width / (float)nchwInput.w;	// 还原系数
				float hCoeff = recRoi.height / (float)nchwInput.h;
				
				for (int v = 0; v < nCountPerBatch - 1; v += 2)
				{
					int offset = nCountPerBatch * u + v;	// 坐标值偏移量

					retTemp[offset] = retTemp[offset] * wCoeff + recRoi.x;
					retTemp[offset + 1] = retTemp[offset + 1] * hCoeff + recRoi.y;
				}
			}
		}

		// save
		vecOutputs.push_back(retTemp);
	}

	// unlock
	//m_bRunningLock = false;

	return RKNN_SUCC;
}
