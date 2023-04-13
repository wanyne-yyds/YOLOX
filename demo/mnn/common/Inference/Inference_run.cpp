#include "Inference.h"

int BSJ_AI::Inference::resizeImage(const ImageData &src, CV::Mat &image) {
    image.release();
    CV::Mat matImage;
    switch (src.format) {
    case IMAGE_FORMAT::BGR888:
    case IMAGE_FORMAT::RGB888:
        matImage = CV::Mat(src.imgHeight, src.imgWidth, 3, (unsigned char *)src.data);
        if (src.imgWidth != m_stCfg.netWidth || src.imgHeight != m_stCfg.netHeight) {
            CV::resize(matImage, image, Size(m_stCfg.netWidth, m_stCfg.netHeight), m_stCfg.filterType);
        } else {
            image = matImage;
        }
        break;
    case IMAGE_FORMAT::GRAY:
        matImage = CV::Mat(src.imgHeight, src.imgWidth, 1, (unsigned char *)src.data);
        if (src.imgWidth != m_stCfg.netWidth || src.imgHeight != m_stCfg.netHeight) {
            CV::resize(matImage, image, Size(m_stCfg.netWidth, m_stCfg.netHeight), m_stCfg.filterType);
        } else {
            image = matImage;
        }
        break;
    case IMAGE_FORMAT::NV12:
    case IMAGE_FORMAT::NV21:
        matImage = CV::Mat(src.imgHeight + (src.imgHeight >> 1), src.imgWidth, 1, (unsigned char *)src.data);
        if (src.imgWidth != m_stCfg.netWidth || src.imgHeight != m_stCfg.netHeight) {
            CV::resizeYUV420sp(matImage, image, Size(m_stCfg.netWidth, m_stCfg.netHeight), m_stCfg.filterType);
        } else {
            image = matImage;
        }
        break;
    default:
        LOGE("BSJ_AI::Inference::run not support image format %d\n", src.format);
        return BSJ_AI_FLAG_FAILED;
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::Inference::convertImage(const CV::Mat &src, CV::Mat &dst, IMAGE_FORMAT format) {
    // 转换数据
    switch (m_stCfg.forward_type) {
    case InferenceType::FORWARD_NCNN:
    case InferenceType::FORWARD_MNN:
    case InferenceType::FORWARD_ROCKCHIP:
        if (format == IMAGE_FORMAT::NV12) {
            CV::cvtColor(src, dst, CV::ColorConversionType::COLOR_CONVERT_NV12TOBGR);
        } else if (format == IMAGE_FORMAT::NV21) {
            CV::cvtColor(src, dst, CV::ColorConversionType::COLOR_CONVERT_NV21TOBGR);
        } else if (format == IMAGE_FORMAT::RGB888) {
            CV::cvtColor(src, dst, CV::ColorConversionType::COLOR_CONVERT_RGBTOBGR);
        } else {
            dst = src;
        }
        break;
    default:
        break;
    }
}

int BSJ_AI::Inference::run(const ImageData &inputData, CallBack &outputCallBack) {
    BSJ_AI_MODEL_FORWARD_ASSERT(inputData.data == NULL, "inputData.data == NULL");
    BSJ_AI_MODEL_FORWARD_ASSERT(inputData.imgWidth == 0, "inputData.imgWidth == 0");
    BSJ_AI_MODEL_FORWARD_ASSERT(inputData.imgHeight == 0, "inputData.imgHeight == 0");
    BSJ_AI_MODEL_FORWARD_ASSERT(inputData.format != m_stCfg.srcFormat, "inputData.format != m_stCfg.srcFormat");

    // 智能锁 一直等待（阻塞）
    std::unique_lock<std::mutex> _l(lock);

    // 压缩图像
    int flag = BSJ_AI_FLAG_SUCCESSFUL;

    CV::Mat image;
    flag = this->resizeImage(inputData, image);
    if (flag != BSJ_AI_FLAG_SUCCESSFUL) {
        return flag;
    }

    // 转换数据
    CV::Mat matImage;
    this->convertImage(image, matImage, inputData.format);

    // 推理
    switch (m_stCfg.forward_type) {
    case InferenceType::FORWARD_NCNN:
        flag = this->run_ncnn(matImage, outputCallBack);
        break;
    case InferenceType::FORWARD_MNN:
        flag = this->run_mnn(matImage, outputCallBack);
        break;
    case InferenceType::FORWARD_ROCKCHIP:
        flag = this->run_rockchip(matImage, outputCallBack);
        break;
    case InferenceType::FORWARD_SIGMASTAR:
        flag = this->run_sigmastar(image, outputCallBack);
        break;
    default:
        break;
    }

    return flag;
}

int BSJ_AI::Inference::run(const ImageData &inputData, std::vector<std::vector<float>> &vecOutputs) {
    vecOutputs.clear();

    CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<NCHW> &shapes) {
        if (shapes.size() == 0) {
            return;
        }

        std::vector<float> output;
        for (int i = 0; i < outputs.size(); i++) {
            for (int j = 0; j < shapes[i].size(); j++) {
                output.push_back(outputs[i][j]);
            }
        }
        vecOutputs.push_back(output);
    };

    // 推理
    return this->run(inputData, func_call_back);
}

int BSJ_AI::Inference::run(const ImageData &inputData, std::vector<float> &vecOutputs) {
    vecOutputs.clear();

    CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<NCHW> &shapes) {
        if (shapes.size() == 0) {
            return;
        }

        for (int j = 0; j < shapes[0].size(); j++) {
            vecOutputs.push_back(outputs[0][j]);
        }
    };

    // 推理
    return this->run(inputData, func_call_back);
}

int BSJ_AI::Inference::run(const ImageData &inputData, float &score, int &label) {
    BSJ_AI_MODEL_FORWARD_ASSERT(m_stCfg.sOupNodes.size() != 1 && m_stCfg.oupNodes.size() != 1, "only support sOupNodes.size() == 1 || m_stCfg.oupNodes.size() == 1")

    CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<NCHW> &shapes) {
        if (shapes.size() == 0) {
            return;
        }

        score = outputs[0][0];
        for (int i = 1; i < shapes[0].size(); i++) {
            float prob = outputs[0][i];
            if (prob > score) {
                score = prob;
                label = i;
            }
        }
    };

    // 推理
    return this->run(inputData, func_call_back);
}
