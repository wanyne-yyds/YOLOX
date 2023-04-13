#include "yolox.h"
#include <cmath>
int BSJ_AI::YOLOX::init(const Config &cfg) {
    return this->baseInit(cfg, "yolox");
}

int BSJ_AI::YOLOX::detect(const ImageData &inputData, std::vector<Object> &vecObjects) {
    vecObjects.clear();

    // gather all box
    std::vector<Object> all_bbox_rects;
    std::vector<float>  all_bbox_scores;

    BSJ_AI::Inference::CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<BSJ_AI::Inference::NCHW> &shapes) {
        if (shapes.size() == 0 || outputs.size() != 3) {
            LOGE("BSJ_AI::YOLOX::detect  shapes.size() == 0 || outputs.size()[%d] != 3)\n", int(outputs.size()));
            return;
        }
        int i = index % m_stCfg.baseCfg.strides.size();
        if (index < m_stCfg.baseCfg.strides.size()) {
            // reg, obj, cls
            this->decode(outputs[0], outputs[1], outputs[2], m_stCfg.baseCfg.strides[i], all_bbox_rects, all_bbox_scores);
        } else {
            // face reg, face obj, face cls
            this->FaceDecode(outputs[0], outputs[1], outputs[2], m_stCfg.baseCfg.strides[i], all_bbox_rects, all_bbox_scores);
        }
    };

    int nResult = m_hForward->run(inputData, func_call_back);
    this->postprocess(inputData, vecObjects, all_bbox_rects, all_bbox_scores);

    return 0;
}

void BSJ_AI::YOLOX::decode(float *preds_reg, float *preds_obj, float *preds_cls, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    int num_grid_x = ceil(m_stCfg.baseCfg.netWidth / stride);
    int num_grid_y = ceil(m_stCfg.baseCfg.netHeight / stride);
    // LOGE("\n");
    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            int   p        = i * num_grid_x + j;
            float box_prob = preds_obj[p];
            // LOGE("%.4f ", box_prob);
            if (box_prob < m_stCfg.baseCfg.thresh)
                continue;
            for (int class_idx = 0; class_idx < m_stCfg.baseCfg.nClasses; class_idx++) {
                float box_cls_score = preds_cls[p + class_idx * num_grid_x * num_grid_y];
                float score         = box_prob * box_cls_score;
                if (score < m_stCfg.baseCfg.thresh)
                    continue;
                // class loop
                float x_center = (preds_reg[p + 0 * num_grid_x * num_grid_y] + j) * stride;
                float y_center = (preds_reg[p + 1 * num_grid_x * num_grid_y] + i) * stride;
                float w        = std::exp(preds_reg[p + 2 * num_grid_x * num_grid_y]) * stride;
                float h        = std::exp(preds_reg[p + 3 * num_grid_x * num_grid_y]) * stride;
                float x0       = x_center - w * 0.5f;
                float y0       = y_center - h * 0.5f;
                // LOGE("x = %.2f, y = %.2f, w = %.2f, h = %.2f, s = %.6f, label = %d \n", x0, y0, w, h, box_prob, class_idx);

                Object obj;
                obj.xmin  = x0;
                obj.ymin  = y0;
                obj.xmax  = x0 + w;
                obj.ymax  = y0 + h;
                obj.label = class_idx;
                obj.score = score;

                all_bbox_rects.push_back(obj);
                all_bbox_scores.push_back(score);
            }
        }
    }
}

void BSJ_AI::YOLOX::FaceDecode(float *preds_reg, float *preds_obj, float *preds_cls, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    int num_grid_x = ceil(m_stCfg.baseCfg.netWidth / stride);
    int num_grid_y = ceil(m_stCfg.baseCfg.netHeight / stride);
    int nClasses   = 2;

    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            int   p        = i * num_grid_x + j;
            float box_prob = preds_obj[p];
            if (box_prob < m_stCfg.baseCfg.thresh)
                continue;
            for (int class_idx = 0; class_idx < nClasses; class_idx++) {
                float box_cls_score = preds_cls[p + class_idx * num_grid_x * num_grid_y];
                float score         = box_prob * box_cls_score;
                if (score < m_stCfg.baseCfg.thresh)
                    continue;
                // class loop
                float x_center = (preds_reg[p + 0 * num_grid_x * num_grid_y] + j) * stride;
                float y_center = (preds_reg[p + 1 * num_grid_x * num_grid_y] + i) * stride;
                float w        = std::exp(preds_reg[p + 2 * num_grid_x * num_grid_y]) * stride;
                float h        = std::exp(preds_reg[p + 3 * num_grid_x * num_grid_y]) * stride;
                float x0       = x_center - w * 0.5f;
                float y0       = y_center - h * 0.5f;
                // LOGE("x = %.2f, y = %.2f, w = %.2f, h = %.2f, s = %.6f, label = %d \n", x0, y0, w, h, box_prob, class_idx);

                Object obj;
                obj.xmin  = x0;
                obj.ymin  = y0;
                obj.xmax  = x0 + w;
                obj.ymax  = y0 + h;
                obj.label = class_idx + m_stCfg.baseCfg.nClasses;
                obj.score = score;

                all_bbox_rects.push_back(obj);
                all_bbox_scores.push_back(score);
            }
        }
    }
}
