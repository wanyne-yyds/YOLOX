#include "Inference.h"

BSJ_AI::Inference::Inference() {
    this->create_rockchip();
}

BSJ_AI::Inference::~Inference() {
    this->destory();
}

void BSJ_AI::Inference::destory() {
    // ������ һֱ�ȴ���������
    std::unique_lock<std::mutex> _l(lock);

    this->destory_mnn();
    this->destory_ncnn();
    this->destory_rockchip();
    this->destory_sigmastar();
}
