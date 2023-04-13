#include "ssd.h"
#include <cmath>

int BSJ_AI::SSD::init(const Config &cfg) {
    int nResult = this->baseInit(cfg, "ssd");
    if (nResult != BSJ_AI_FLAG_SUCCESSFUL) {
        return nResult;
    }
    BSJ_AI_MODEL_FORWARD_ASSERT(m_stCfg.baseCfg.strides.size() != m_stCfg.anchors.size(), "anchors.size() != strides.size()");
    // 生成候选框 anchor base
    this->PriorBox();
    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::SSD::PriorBox() {
    // 只需要第一次生成
    for (int i = 0; i < m_stCfg.baseCfg.strides.size(); i++) {
        std::vector<float> min_size;
        min_size = m_stCfg.anchors[i]; // 每个特征图对应的anchor base

        int rows = ceil(float(m_stCfg.baseCfg.netHeight) / m_stCfg.baseCfg.strides[i]); // 行，高
        int cols = ceil(float(m_stCfg.baseCfg.netWidth) / m_stCfg.baseCfg.strides[i]);  // 列，宽

        // 训练所采用的是 chw，故输出也是hw, 计算以行优先
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // 每个特征图有两个anchor base
                for (int j = 0; j < min_size.size(); j++) {
                    BSJ_AI::Rect2f box;
                    float          s_kx = float(min_size[j]) / m_stCfg.baseCfg.netWidth;  // 宽 比例 归一化
                    float          s_ky = float(min_size[j]) / m_stCfg.baseCfg.netHeight; // 高 比例 归一化

                    // float(_steps[i]) / _imageWidth 表示_feature_map大小 也可以直接除以 cols，表示x相对于特征图的比例
                    float dense_cx = (col + 0.5) * float(m_stCfg.baseCfg.strides[i]) / m_stCfg.baseCfg.netWidth;
                    float dense_cy = (row + 0.5) * float(m_stCfg.baseCfg.strides[i]) / m_stCfg.baseCfg.netHeight;

                    box.x      = dense_cx;
                    box.y      = dense_cy;
                    box.width  = s_kx;
                    box.height = s_ky;

                    m_vDefaultBoxs.push_back(box);
                    // std::cout << box.x << "\t" << box.y << "\t" << box.w << "\t" << box.h << std::endl;
                }
            }
        }
    }
}

int BSJ_AI::SSD::detect(const ImageData &inputData, std::vector<Object> &vecObjects) {
    vecObjects.clear();

    // gather all box
    std::vector<Object> all_bbox_rects;
    std::vector<float>  all_bbox_scores;

    BSJ_AI::Inference::CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<BSJ_AI::Inference::NCHW> &shapes) {
        if (shapes.size() == 0 || outputs.size() != 2) {
            LOGE("BSJ_AI::SSD::detect  shapes.size() == 0 || outputs.size()[%d] != 3)\n", int(outputs.size()));
            return;
        }
        this->decode(outputs[0], outputs[1], all_bbox_rects, all_bbox_scores);
    };

    int nResult = m_hForward->run(inputData, func_call_back);

    this->postprocess(inputData, vecObjects, all_bbox_rects, all_bbox_scores);
    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::SSD::decode(float *score_blob, float *bbox_blob, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    for (int i = 0; i < m_vDefaultBoxs.size(); i++) {
        float score = score_blob[2 * i + 1];

        if (score < m_stCfg.baseCfg.thresh) {
            continue;
        }

        BSJ_AI::Rect2f anchor = m_vDefaultBoxs[i];

        // apply center size bbox
        float dx = bbox_blob[i * 4 + 0]; // x center
        float dy = bbox_blob[i * 4 + 1]; // y center
        float dw = bbox_blob[i * 4 + 2]; // w
        float dh = bbox_blob[i * 4 + 3]; // h

        // std::cout << dx << " " << dy << " " << dw << " " << dh << std::endl;

        float bbox_cx = anchor.x + dx * m_stCfg.variances[0] * anchor.width; //
        float bbox_cy = anchor.y + dy * m_stCfg.variances[1] * anchor.height;
        float bbox_w  = anchor.width * exp(m_stCfg.variances[2] * dw);
        float bbox_h  = anchor.height * exp(m_stCfg.variances[3] * dh);

        float x0 = (bbox_cx - bbox_w * 0.5f) * m_stCfg.baseCfg.netWidth;
        float y0 = (bbox_cy - bbox_h * 0.5f) * m_stCfg.baseCfg.netHeight;
        float x1 = (bbox_cx + bbox_w * 0.5f) * m_stCfg.baseCfg.netWidth;
        float y1 = (bbox_cy + bbox_h * 0.5f) * m_stCfg.baseCfg.netHeight;

        // std::cout << x0 << " " << y0 << " " <<x1 << " " << y1 << std::endl;

        Object bbox;
        bbox.xmin  = x0;
        bbox.ymin  = y0;
        bbox.xmax  = x1;
        bbox.ymax  = y1;
        bbox.label = 0;
        bbox.score = score;
        all_bbox_scores.push_back(score);
        all_bbox_rects.push_back(bbox);
    }
}