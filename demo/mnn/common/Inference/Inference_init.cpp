#include "Inference.h"

int BSJ_AI::Inference::init(const Config &cfg) {
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.netWidth == 0, "cfg.netWidth == 0");
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.netHeight == 0, "cfg.netHeight == 0");

    this->destory();

    // 智能锁 一直等待（阻塞）
    // std::unique_lock<std::mutex> _l(lock);

    int flag = 0;
    switch (cfg.forward_type) {
    case InferenceType::FORWARD_NCNN:
        flag = this->init_ncnn(cfg);
        break;
    case InferenceType::FORWARD_MNN:
        flag = this->init_mnn(cfg);
        break;
    case InferenceType::FORWARD_ROCKCHIP:
        flag = this->init_rockchip(cfg);
        break;
    case InferenceType::FORWARD_SIGMASTAR:
        flag = this->init_sigmastar(cfg);
        break;
    default:
        break;
    }

    if (flag != BSJ_AI_FLAG_SUCCESSFUL) {
        this->destory();
    }

    // 赋值
    m_stCfg = cfg;
    return flag;
}
