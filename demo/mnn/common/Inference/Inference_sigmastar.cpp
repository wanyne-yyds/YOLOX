#include "Inference.h"

void BSJ_AI::Inference::create_sigmastar() {
}

#ifdef USE_SIGMASTAR
MI_S32 IPU_Free(MI_IPU_Tensor_t *pTensor, MI_U32 BufSize) {
    MI_S32 s32ret = 0;
    s32ret        = MI_SYS_Munmap(pTensor->ptTensorData[0], BufSize);
    s32ret        = MI_SYS_MMA_Free(pTensor->phyTensorAddr[0]);
    return s32ret;
}
static int H2SerializedReadFunc_2(void *dst_buf, int offset, int size, char *ctx) {
    // read data from buf
    LOGE("read from call back function\n");
    memcpy(dst_buf, ctx + offset, size);
    return 0;
}
#endif

void BSJ_AI::Inference::destory_sigmastar() {
#ifdef USE_SIGMASTAR
    // 释放当前通道
    if (m_hU32ChannelID != 65535) {
        MI_IPU_DestroyCHN(m_hU32ChannelID);

        IPU_Free(&m_InputTensorVector.astArrayTensors[0], m_nDesc.astMI_InputTensorDescs[0].s32AlignedBufSize);
        for (MI_S32 idx = 0; idx < m_nDesc.u32OutputTensorCount; idx++) {
            IPU_Free(&m_OutputTensorVector.astArrayTensors[idx], m_nDesc.astMI_OutputTensorDescs[idx].s32AlignedBufSize);
        }
    }

#endif // USE_SIGMASTAR
}

int BSJ_AI::Inference::init_sigmastar(const Config &cfg) {
#ifdef USE_SIGMASTAR
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.sInpNodes.size() != 1 || cfg.sOupNodes.empty(), "cfg.sInpNodes.size() != 1 || cfg.sOupNodes is empty");

    // 巧用析构函数
    BSJ_AI::defer<void(void)> init_func([&](void) {
        this->destory_sigmastar();
    });

    MI_S32 s32Ret = MI_SUCCESS;
    // 1. 创建 IPU 通道
    MI_SYS_GlobalPrivPoolConfig_t stGlobalPrivPoolConf;
    MI_IPUChnAttr_t               stChnAttr;

    memset(&stChnAttr, 0, sizeof(stChnAttr));
    // 输入
    if (cfg.use_zero_copy) {
        stChnAttr.u32InputBufDepth = 0;
    } else {
        stChnAttr.u32InputBufDepth = 1;
    }
    // 输出
    stChnAttr.u32OutputBufDepth = 1;
    char str[256];

    // 加载模型
    if (cfg.model && cfg.nModelSize) {
        // 头文件形式加载
        s32Ret = MI_IPU_CreateCHN(&m_hU32ChannelID, &stChnAttr, H2SerializedReadFunc_2, (char *)cfg.model);
        LOGE("BSJ_AI::DLA::init load model from model header file");
    } else {
        // 模型文件加载
        BSJ_AI_MODEL_FORWARD_ASSERT(cfg.model_path.empty(), "cfg.model_path is empty");
        s32Ret = MI_IPU_CreateCHN(&m_hU32ChannelID, &stChnAttr, NULL, (char *)cfg.model_path.data());
        LOGE("BSJ_AI::Inference::init_sigmastar load model from %s\n", cfg.model_path.c_str());
    }

    sprintf(str, "BSJ_AI::Inference::init_sigmastar err: MI_IPU_CreateCHN failed with %d.", s32Ret);
    BSJ_AI_MODEL_FORWARD_ASSERT(s32Ret != MI_SUCCESS, str);

    // 2. 获取模型输入输出 Tensor 属性
    MI_IPU_GetInOutTensorDesc(m_hU32ChannelID, &m_nDesc);

    MI_IPU_ELEMENT_FORMAT eElmFormat;
    // 打印输入信息
    for (int i = 0; i < m_nDesc.u32InputTensorCount; i++) {
        eElmFormat = m_nDesc.astMI_InputTensorDescs[i].eElmFormat;
        LOGE("InputTensorDescs index = %d, u32TensorDim: %d\n", i, m_nDesc.astMI_InputTensorDescs[i].u32TensorDim);
        LOGE("InputTensorDescs index = %d, eElmFormat: %d\n", i, m_nDesc.astMI_InputTensorDescs[i].eElmFormat);
        LOGE("InputTensorDescs index = %d, u32InnerMostStride: %d\n", i, m_nDesc.astMI_InputTensorDescs[i].u32InnerMostStride);
        LOGE("InputTensorDescs index = %d, fScalar: %d\n", i, m_nDesc.astMI_InputTensorDescs[i].fScalar);
        LOGE("InputTensorDescs index = %d, s64ZeroPoint: %d\n", i, m_nDesc.astMI_InputTensorDescs[i].s64ZeroPoint);
        LOGE("InputTensorDescs index = %d, s32AlignedBufSize: %d\n", i, m_nDesc.astMI_InputTensorDescs[i].s32AlignedBufSize);
        // nhwc
        for (int j = 0; j < m_nDesc.astMI_InputTensorDescs[i].u32TensorDim; j++) {
            LOGE("InputTensorDescs index = %d u32TensorShape index = %d shape = %d \n", i, j, m_nDesc.astMI_InputTensorDescs[i].u32TensorShape[j]);
        }
    }

    // 打印输出信息
    for (int i = 0; i < m_nDesc.u32OutputTensorCount; i++) {
        LOGE("OutputTensorDescs index = %d, u32TensorDim: %d\n", i, m_nDesc.astMI_OutputTensorDescs[i].u32TensorDim);
        LOGE("OutputTensorDescs index = %d, eElmFormat: %d\n", i, m_nDesc.astMI_OutputTensorDescs[i].eElmFormat);
        LOGE("OutputTensorDescs index = %d, u32InnerMostStride: %d\n", i, m_nDesc.astMI_OutputTensorDescs[i].u32InnerMostStride);
        LOGE("OutputTensorDescs index = %d, fScalar: %d\n", i, m_nDesc.astMI_OutputTensorDescs[i].fScalar);
        LOGE("OutputTensorDescs index = %d, s64ZeroPoint: %d\n", i, m_nDesc.astMI_OutputTensorDescs[i].s64ZeroPoint);
        LOGE("OutputTensorDescs index = %d, s32AlignedBufSize: %d\n", i, m_nDesc.astMI_OutputTensorDescs[i].s32AlignedBufSize);
        // nhwc
        for (int j = 0; j < m_nDesc.astMI_OutputTensorDescs[i].u32TensorDim; j++) {
            LOGE("OutputTensorDescs index = %d u32TensorShape index = %d shape = %d \n", i, j, m_nDesc.astMI_OutputTensorDescs[i].u32TensorShape[j]);
        }
    }

    int nModelHeight  = m_nDesc.astMI_InputTensorDescs[0].u32TensorShape[1];
    int nModelWidth   = m_nDesc.astMI_InputTensorDescs[0].u32TensorShape[2];
    int nModelChannel = m_nDesc.astMI_InputTensorDescs[0].u32TensorShape[3];

    sprintf(str, "BSJ_AI::Inference::init_sigmastar err: nModelWidth[%d] != cfg.nModelWidth[%d]", nModelWidth, cfg.netWidth);
    BSJ_AI_MODEL_FORWARD_ASSERT(nModelWidth != cfg.netWidth, str);
    sprintf(str, "BSJ_AI::Inference::init_sigmastar err: nModelHeight[%d] != cfg.netHeight[%d]", nModelHeight, cfg.netHeight);
    BSJ_AI_MODEL_FORWARD_ASSERT(nModelHeight != cfg.netHeight, str);

    // 输入
    if (cfg.use_zero_copy) {
        m_InputTensorVector.u32TensorCount = m_nDesc.u32InputTensorCount;
        LOGE("BSJ_AI::Inference::init_sigmastar load MI, use zero copy\n");
    } else {
        s32Ret = MI_IPU_GetInputTensors(m_hU32ChannelID, &m_InputTensorVector);
        sprintf(str, "BSJ_AI::Inference::init_sigmastar err: MI_IPU_GetInputTensors %d", s32Ret);
        BSJ_AI_MODEL_FORWARD_ASSERT(s32Ret != MI_SUCCESS, str);
        LOGE("BSJ_AI::Inference::init_sigmastar not load MI\n");
    }

    // 输出
    s32Ret = MI_IPU_GetOutputTensors(m_hU32ChannelID, &m_OutputTensorVector);
    sprintf(str, "BSJ_AI::Inference::init_sigmastar err: MI_IPU_GetOutputTensors %d", s32Ret);
    BSJ_AI_MODEL_FORWARD_ASSERT(s32Ret != MI_SUCCESS, str);

    if (cfg.srcFormat == IMAGE_FORMAT::NV12 && eElmFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_NV12) {
        LOGE("BSJ_AI::DLA::init InputTensorDescs eElmFormat is NV12\n");
    } else if (cfg.srcFormat == IMAGE_FORMAT::BGR888 && eElmFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_U8) {
        LOGE("BSJ_AI::DLA::init InputTensorDescs eElmFormat is BGR\n");
    } else if (cfg.srcFormat == IMAGE_FORMAT::RGB888 && eElmFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_U8) {
        LOGE("BSJ_AI::DLA::init InputTensorDescs eElmFormat is RGB\n");
    } else {
        LOGE("BSJ_AI::DLA::init InputTensorDescs eElmFormat[%d] != m_nFormat[%d]\n", eElmFormat, cfg.srcFormat);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    // 取消
    init_func.cancel_call();
#else
    LOGE("please use sigmastar\n");
    return BSJ_AI_FLAG_FAILED;
#endif // USE_SIGMASTAR
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int BSJ_AI::Inference::run_sigmastar(const CV::Mat &image, CallBack &outputCallBack) {
#ifdef USE_SIGMASTAR
    BSJ_AI_MODEL_FORWARD_ASSERT(image.empty(), "image is empty");
    BSJ_AI_MODEL_FORWARD_ASSERT(m_hU32ChannelID == 65535, "run sigmastar init err,m_hU32ChannelID == 65535");

    int imageSize = image.rows * image.cols * image.channels;
    // 输入数据
    // 1.来自mi模块
    if (m_stCfg.use_zero_copy) {
        // m_InputTensorVector.astArrayTensors[0].phyTensorAddr[0] = imageData.pstBufInfo->stFrameData.phyAddr[0];
        // m_InputTensorVector.astArrayTensors[0].ptTensorData[0] = imageData.pstBufInfo->stFrameData.pVirAddr[0];
        // m_InputTensorVector.astArrayTensors[0].phyTensorAddr[1] = imageData.pstBufInfo->stFrameData.phyAddr[1];
        // m_InputTensorVector.astArrayTensors[0].ptTensorData[1] = imageData.pstBufInfo->stFrameData.pVirAddr[1];
    } else {
        memcpy(m_InputTensorVector.astArrayTensors[0].ptTensorData[0], (unsigned char *)image.data, imageSize);
        // memset(m_InputTensorVector.astArrayTensors[0].ptTensorData[0], 0, imageSize);
        MI_SYS_FlushInvCache(m_InputTensorVector.astArrayTensors[0].ptTensorData[0], imageSize);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  模型推演
    uint64_t t1     = BSJ_AI::getTickMillitm();
    MI_S32   s32Ret = MI_IPU_Invoke(m_hU32ChannelID, &m_InputTensorVector, &m_OutputTensorVector);
    char     str[256];
    sprintf(str, "BSJ_AI::Inference::run_sigmastar err: MI_IPU_Invoke %d", s32Ret);
    BSJ_AI_MODEL_FORWARD_ASSERT(s32Ret != MI_SUCCESS, str);

    // 获取输出
    int index = 0;
    for (int i = 0; i < m_stCfg.sOupNodes.size(); i++) {
        int                  nSize = m_stCfg.sOupNodes[i].size();
        std::vector<float *> outputs;
        std::vector<NCHW>    shapes;
        outputs.resize(nSize);

        for (int j = 0; j < nSize; j++) {
            float *output = (float *)m_OutputTensorVector.astArrayTensors[index].ptTensorData[0];

            /*
             * Note: 一般我们使用nhwc进行输入，那么输出必定是nhwc
             * 由于astMI_OutputTensorDescs获取不到输入输出为nchw还是nhwc，因此转模型的时候需要注意
             * 解码用的是chw, 所以hwc要转成chw
             */
            BSJ_AI::Inference::NCHW nchwOuput = BSJ_AI::Inference::NCHW(1, 1, 1, 1);

            int lastIndex = m_nDesc.astMI_OutputTensorDescs[index].u32TensorDim - 1;
            nchwOuput.n   = m_nDesc.astMI_OutputTensorDescs[index].u32TensorShape[0];
            nchwOuput.c   = m_nDesc.astMI_OutputTensorDescs[index].u32TensorShape[lastIndex] ? m_nDesc.astMI_OutputTensorDescs[index].u32TensorShape[lastIndex] : 1;
            nchwOuput.h   = lastIndex > 1 ? m_nDesc.astMI_OutputTensorDescs[index].u32TensorShape[1] : 1;
            nchwOuput.w   = lastIndex > 2 ? m_nDesc.astMI_OutputTensorDescs[index].u32TensorShape[2] : 1;

            int dataLenght = nchwOuput.size();

            /*
             * Note: 注意8字节对齐
             * 如果输出的c通道为8的倍数，那么输出不需要进行补齐
             * 例如:  输出的c为16，则maxSize = 2， 排布为 (0 1 2 ... 6 7 8) (0 1 2 ... 6 7 8)
             *       输出的c为18，则maxSize = 3， 排布为 (0 1 2 ... 6 0 0) (0 1 2 ... 6 0 0) (0 1 2 ... 6 0 0)
             */
            int maxSize = nchwOuput.c % 8 ? nchwOuput.c / 8 + 1 : nchwOuput.c / 8;
            outputs[j]  = (float *)malloc(sizeof(float) * dataLenght);

            int hw = nchwOuput.h * nchwOuput.w;
            for (int y = 0; y < nchwOuput.h; y++) {
                for (int x = 0; x < nchwOuput.w; x++) {
                    int yx = y * nchwOuput.w + x;
                    for (int c = 0; c < nchwOuput.c; c++) {
                        outputs[j][c * hw + yx] = output[yx * maxSize * 8 + c];
                    }
                }
            }

            // nchw
            shapes.push_back(nchwOuput);

            index++;
        }

        outputCallBack(i, outputs, shapes);
        for_each(outputs.begin(), outputs.end(), [](float *&iter) {if (iter != NULL) { delete iter; iter = nullptr; } });
    }
#else
    LOGE("please use sigmastar\n");
    return BSJ_AI_FLAG_FAILED;
#endif // USE_SIGMASTAR
    return BSJ_AI_FLAG_SUCCESSFUL;
}