#include "Inference.h"

void BSJ_AI::Inference::create_mnn() {
}

void BSJ_AI::Inference::destory_mnn() {
#ifdef USE_MNN
    if (m_hMnn) {
        m_hMnn->releaseSession(session);
    }
#endif
}

int BSJ_AI::Inference::init_mnn(const Config &cfg) {
#ifdef USE_MNN
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.model_path.empty() && (!cfg.model || !cfg.nModelSize), "cfg.model_path && (!cfg.model || !cfg.nModelSize) is empty");
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.sInpNodes.size() != 1 || cfg.sOupNodes.empty(), "cfg.sInpNodes.size() != 1 || cfg.sOupNodes is empty");
    // 加载模型
    if (cfg.model) {
        m_hMnn.reset(MNN::Interpreter::createFromBuffer((void *)cfg.model, cfg.nModelSize));
    } else {
        m_hMnn.reset(MNN::Interpreter::createFromFile(cfg.model_path.c_str()));
    }

    BSJ_AI_MODEL_FORWARD_ASSERT(m_hMnn == nullptr, "m_hMnn is nullptr");

    // 调度配置
    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    config.numThread = cfg.nThread;

    //// 后端配置
    MNN::BackendConfig backendConfig;
    backendConfig.memory    = MNN::BackendConfig::MemoryMode::Memory_Normal;
    backendConfig.power     = MNN::BackendConfig::PowerMode::Power_Normal;
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
    config.backendConfig    = &backendConfig;

    // 创建session
    session = m_hMnn->createSession(config);

    BSJ_AI_MODEL_FORWARD_ASSERT(session == NULL, "session == NULL");
#else
    LOGE("please use mnn\n");
    return -1;
#endif // USE_MNN
    return 0;
}

#ifdef USE_MNN
int convertImageMNN(BSJ_AI::IMAGE_FORMAT format) {
    switch (format) {
    case BSJ_AI::IMAGE_FORMAT::RGB888:
        return MNN::CV::ImageFormat::RGB;
    case BSJ_AI::IMAGE_FORMAT::BGR888:
        return MNN::CV::ImageFormat::BGR;
    case BSJ_AI::IMAGE_FORMAT::NV21:
        return MNN::CV::ImageFormat::YUV_NV21;
    case BSJ_AI::IMAGE_FORMAT::NV12:
        return MNN::CV::ImageFormat::YUV_NV12;
    default:
        break;
    }
    return -1;
}
#endif // USE_MNN
int BSJ_AI::Inference::run_mnn(const CV::Mat &image, CallBack &outputCallBack) {
#ifdef USE_MNN
    BSJ_AI_MODEL_FORWARD_ASSERT(image.empty(), "image is empty");
    BSJ_AI_MODEL_FORWARD_ASSERT(m_hMnn == NULL, "m_hMnn == NULL");
    BSJ_AI_MODEL_FORWARD_ASSERT(session == NULL, "session == NULL");

    uint64_t tt = BSJ_AI::getTickMillitm();

    auto input_tensor = m_hMnn->getSessionInput(session, nullptr);
    auto input_shape  = input_tensor->shape();

    m_hMnn->resizeTensor(input_tensor, input_shape); // nchw
    m_hMnn->resizeSession(session);

    // LOGI("resizeSession %lu ms \n", BSJ_AI::getTickMillitm() - tt);

    tt = BSJ_AI::getTickMillitm();

    // 图像处理
    MNN::CV::ImageProcess::Config config;
    config.filterType = MNN::CV::BILINEAR; // 缩放方式
    ::memcpy(config.mean, m_stCfg.mean, sizeof(m_stCfg.mean));
    ::memcpy(config.normal, m_stCfg.normal, sizeof(m_stCfg.normal));

    // 格式转换
    switch (image.channels) {
    case 3:
        config.sourceFormat = MNN::CV::ImageFormat::BGR;
        config.destFormat   = MNN::CV::ImageFormat::BGR;
        break;
    case 1:
        config.sourceFormat = MNN::CV::ImageFormat::GRAY;
        config.destFormat   = MNN::CV::ImageFormat::GRAY;
        break;
    default:
        LOGE("BSJ_AI::Inference::run_mnn image channels is not 1 or 3!");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }
    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    MNN::ErrorCode                         errCode = process->convert((uint8_t *)image.data, m_stCfg.netWidth, m_stCfg.netHeight, 0, input_tensor);

    // LOGI("ImageProcess %lu ms \n", BSJ_AI::getTickMillitm() - tt);

    // 运行会话
    uint64_t t1 = BSJ_AI::getTickMillitm();
    errCode     = m_hMnn->runSession(session);

    // LOGI("runSession %lu ms \n", BSJ_AI::getTickMillitm() - t1);

    // 获取输出
    for (int i = 0; i < m_stCfg.sOupNodes.size(); i++) {
        int                        nSize = m_stCfg.sOupNodes[i].size();
        std::vector<float *>       outputs;
        std::vector<MNN::Tensor *> tensor_outputs_hosts;
        tensor_outputs_hosts.resize(nSize);

        std::vector<NCHW> shapes;
        for (int j = 0; j < nSize; j++) {
            MNN::Tensor *tensor_outputs = m_hMnn->getSessionOutput(session, m_stCfg.sOupNodes[i][j].c_str());

            tensor_outputs_hosts[j] = MNN::Tensor::create(tensor_outputs->shape(), tensor_outputs->getType(), nullptr, tensor_outputs->getDimensionType());
            tensor_outputs->copyToHostTensor(tensor_outputs_hosts[j]);

            int w = tensor_outputs_hosts[j]->width();
            int h = tensor_outputs_hosts[j]->height();
            int c = tensor_outputs_hosts[j]->channel();

            outputs.push_back(tensor_outputs_hosts[j]->host<float>());
            shapes.push_back(NCHW(1, c, h, w));
        }

        outputCallBack(i, outputs, shapes);

        for_each(tensor_outputs_hosts.begin(), tensor_outputs_hosts.end(), [](MNN::Tensor *&iter) {if(iter!=NULL) {delete iter; iter = nullptr;} });
        // for_each(outputs.begin(), outputs.end(), [](float*& iter) {if(iter!=NULL) {delete[] iter; iter = nullptr;} });
    }
#else
    LOGE("please use mnn\n");
    return -1;
#endif // USE_MNN
    return 0;
}
