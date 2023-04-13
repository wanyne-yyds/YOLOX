#include "Inference.h"

void BSJ_AI::Inference::create_rockchip() {
#ifdef USE_ROCKCHIP
    m_hContext           = 0;
    m_nIoNumber.n_input  = 0;
    m_nIoNumber.n_output = 0;
    m_aInputAttr         = 0;
    m_aInputData         = 0;
    m_aOutputAttr        = 0;
    m_aOutputData        = 0;
#endif // USE_ROCKCHIP
}

void BSJ_AI::Inference::destory_rockchip() {
#ifdef USE_ROCKCHIP
    if (m_hContext) {
        int ret    = rknn_destroy(m_hContext);
        m_hContext = 0;
    }

    if (m_aInputAttr) {
        delete m_aInputAttr;
        m_aInputAttr = 0;
    }
    if (m_aInputData) {
        for (int i = m_nIoNumber.n_input - 1; i >= 0; i--) {
            delete ((uint8_t *)m_aInputData[i].buf);
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
            delete ((uint8_t *)m_aOutputData[i].buf);
        }
        delete m_aOutputData;
        m_aOutputData = 0;
    }

    m_nIoNumber.n_input  = 0;
    m_nIoNumber.n_output = 0;
#endif // USE_ROCKCHIP
}

#ifdef USE_ROCKCHIP
static BSJ_AI::Inference::NCHW getNCHW(const rknn_tensor_attr &tensor) {
    switch (tensor.fmt) {
    case RKNN_TENSOR_NCHW:
        return BSJ_AI::Inference::NCHW(tensor.dims[3], tensor.dims[2], tensor.dims[1], tensor.dims[0]);
    case RKNN_TENSOR_NHWC:
    default:
        return BSJ_AI::Inference::NCHW(tensor.dims[3], tensor.dims[1], tensor.dims[0], tensor.dims[2]);
    }
}

static void printRKNNTensor(rknn_tensor_attr *attr) {
    LOGE("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
         attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
#endif // USE_ROCKCHIP

int BSJ_AI::Inference::init_rockchip(const Config &cfg) {
#ifdef USE_ROCKCHIP
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.model == NULL, "cfg.model == NULL");
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.nModelSize == 0, "cfg.nModelSize == 0");
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.sInpNodes.size() != 1 || cfg.sOupNodes.empty(), "cfg.sInpNodes.size() != 1 || cfg.sOupNodes is empty");

    // 巧用析构函数
    BSJ_AI::defer<void(void)> init_func([&](void) {
        this->destory_rockchip();
    });

    // 加载模型 rknn_init
    int ret = rknn_init(&m_hContext, cfg.model, cfg.nModelSize, RKNN_FLAG_PRIOR_HIGH);

    char str[256];
    sprintf(str, "BSJ_AI::Inference::init_rockchip err: rknn_init failed with %d.", ret);
    BSJ_AI_MODEL_FORWARD_ASSERT(ret != RKNN_SUCC, str);

    // rknn_query io num
    ret = rknn_query(m_hContext, RKNN_QUERY_IN_OUT_NUM, &m_nIoNumber, sizeof(m_nIoNumber));
    // 判断返回值
    sprintf(str, "BSJ_AI::Inference::init_rockchip err: rknn_query(RKNN_QUERY_IN_OUT_NUM) failed with %d.", ret);
    BSJ_AI_MODEL_FORWARD_ASSERT(ret != RKNN_SUCC, str);

    // 仅支持一个输入
    sprintf(str, "BSJ_AI::Inference::init_rockchip err: the number of input tensor is %d. Support if and only if single input tensor!", m_nIoNumber.n_input);
    BSJ_AI_MODEL_FORWARD_ASSERT(m_nIoNumber.n_input != 1, str);

    // rknn_query input attr & malloc rknn_input
    m_aInputAttr = new rknn_tensor_attr[m_nIoNumber.n_input];
    memset(m_aInputAttr, 0, m_nIoNumber.n_input * sizeof(rknn_tensor_attr));
    m_aInputData = new rknn_input[m_nIoNumber.n_input];
    memset(m_aInputData, 0, m_nIoNumber.n_input * sizeof(rknn_input));
    for (int i = 0; i < m_nIoNumber.n_input; i++) {
        m_aInputAttr[i].index = i;
        ret                   = rknn_query(m_hContext, RKNN_QUERY_INPUT_ATTR, &(m_aInputAttr[i]), sizeof(rknn_tensor_attr));

        sprintf(str, "BSJ_AI::Inference::init_rockchip err:  rknn_query(RKNN_QUERY_INPUT_ATTR, %d) failed with %d.", i, ret);
        BSJ_AI_MODEL_FORWARD_ASSERT(ret != RKNN_SUCC, str);

        BSJ_AI::Inference::NCHW nchw = getNCHW(m_aInputAttr[i]);
        // BSJ_AI_MODEL_FORWARD_ASSERT(nchw.c != 3, "BSJ_AI::rknn::init err: the channels of model must be 3!\n");

        // rknn_input
        m_aInputData[i].index        = m_aInputAttr[i].index;
        m_aInputData[i].size         = nchw.size();
        m_aInputData[i].buf          = (void *)new uint8_t[m_aInputData[i].size];
        m_aInputData[i].type         = RKNN_TENSOR_UINT8;
        m_aInputData[i].fmt          = RKNN_TENSOR_NHWC;
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
        ret                    = rknn_query(m_hContext, RKNN_QUERY_OUTPUT_ATTR, &(m_aOutputAttr[i]), sizeof(rknn_tensor_attr));

        sprintf(str, "BSJ_AI::Inference::init_rockchip err: rknn_query(RKNN_QUERY_OUTPUT_ATTR, %d) failed with %d.", i, ret);
        BSJ_AI_MODEL_FORWARD_ASSERT(ret != RKNN_SUCC, str);

        // rknn_output
        m_aOutputData[i].index       = m_aOutputAttr[i].index;
        m_aOutputData[i].want_float  = 1;
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
            LOGE("BSJ_AI::Inference::init_rockchip err: m_aOutputAttr[%d].type = %d.\n", i, m_aOutputAttr[i].type);
            return BSJ_AI_FLAG_BAD_PARAMETER;
        }
        m_aOutputData[i].buf = (void *)new uint8_t[m_aOutputData[i].size];

        printRKNNTensor(&(m_aOutputAttr[i]));
    }
    // 取消
    init_func.cancel_call();
#else
    LOGE("please use rockchip\n");
    return BSJ_AI_FLAG_FAILED;
#endif // USE_ROCKCHIP
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int BSJ_AI::Inference::run_rockchip(const CV::Mat &image, CallBack &outputCallBack) {
#ifdef USE_ROCKCHIP
    BSJ_AI_MODEL_FORWARD_ASSERT(image.empty(), "image is empty");
    BSJ_AI_MODEL_FORWARD_ASSERT(!m_hContext, "BSJ_AI::Inference::run_rockchip err: system is not uninitialized!");

    char str[256];
    sprintf(str, "BSJ_AI::Inference::run_rockchip err: the number of input tensor is %d. Support if and only if single input tensor!\n", m_nIoNumber.n_input);
    BSJ_AI_MODEL_FORWARD_ASSERT(m_nIoNumber.n_input != 1, str);

    // net params
    BSJ_AI::Inference::NCHW nchwInput = getNCHW(m_aInputAttr[0]);
    sprintf(str, "BSJ_AI::Inference::run_rockchip err: nchwInput.w[%d] != cfg.netWidth[%d]", nchwInput.w, m_stCfg.netWidth);
    sprintf(str + strlen(str), "or nchwInput.h[%d] != cfg.netHeight[%d]\n", nchwInput.h, m_stCfg.netHeight);
    BSJ_AI_MODEL_FORWARD_ASSERT(nchwInput.w != m_stCfg.netWidth || nchwInput.h != m_stCfg.netHeight, str);

    int      nResult = BSJ_AI_FLAG_SUCCESSFUL;
    uint64_t t1      = BSJ_AI::getTickMillitm();
    ::memcpy(m_aInputData[0].buf, (uint8_t *)image.data, sizeof(uint8_t) * nchwInput.c * nchwInput.h * nchwInput.w);

    // rknn_inputs_set
    t1      = BSJ_AI::getTickMillitm();
    nResult = rknn_inputs_set(m_hContext, m_nIoNumber.n_input, m_aInputData);
    sprintf(str, "BSJ_AI::Inference::run_rockchip err: err: rknn_input_set fail! ret = %d.", nResult);
    BSJ_AI_MODEL_FORWARD_ASSERT(nResult < 0, str);
    // LOGE("BSJ_AI::rknn::rknn_inputs_set time %lld ms\n", BSJ_AI::getTickMillitm() - t1);

    // 推理
    t1      = BSJ_AI::getTickMillitm();
    nResult = rknn_run(m_hContext, nullptr);
    sprintf(str, "BSJ_AI::Inference::run_rockchip err: err: rknn_run fail! ret = %d.", nResult);
    BSJ_AI_MODEL_FORWARD_ASSERT(nResult < 0, str);
    // LOGE("BSJ_AI::rknn::rknn_run time %lld ms\n", BSJ_AI::getTickMillitm() - t1);

    // 获取所有的检测结果，需提前了解输出顺序和结构
    nResult = rknn_outputs_get(m_hContext, m_nIoNumber.n_output, m_aOutputData, NULL);

    // 获取输出
    int index = 0;
    for (int i = 0; i < m_stCfg.sOupNodes.size(); i++) {
        int                  nSize = m_stCfg.sOupNodes[i].size();
        std::vector<float *> outputs;
        std::vector<NCHW>    shapes;
        for (int j = 0; j < nSize; j++) {
            outputs.push_back((float *)(m_aOutputData[index++].buf));
            shapes.push_back(NCHW(m_aOutputAttr[i].dims[3], m_aOutputAttr[i].dims[2], m_aOutputAttr[i].dims[1], m_aOutputAttr[i].dims[0]));
        }

        outputCallBack(i, outputs, shapes);
    }
#else
    LOGE("please use rockchip\n");
    return BSJ_AI_FLAG_FAILED;
#endif // USE_ROCKCHIP
    return BSJ_AI_FLAG_SUCCESSFUL;
}