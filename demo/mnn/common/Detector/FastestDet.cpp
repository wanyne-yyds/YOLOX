#include "FastestDet.h"
#include <cmath>
int BSJ_AI::FastestDet::init(const Config &cfg) {
    return this->baseInit(cfg, "FastestDet");
}

int BSJ_AI::FastestDet::detect(const ImageData &inputData, std::vector<Object> &vecObjects) {
    vecObjects.clear();

    // gather all box
    std::vector<Object> all_bbox_rects;
    std::vector<float>  all_bbox_scores;

    BSJ_AI::Inference::CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<BSJ_AI::Inference::NCHW> &shapes) {
        if (shapes.size() == 0) {
            return;
        }
        // heatmap
        this->decode(outputs[0], all_bbox_rects, all_bbox_scores);
    };

    int nResult = m_hForward->run(inputData, func_call_back);
    this->postprocess(inputData, vecObjects, all_bbox_rects, all_bbox_scores);
    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::FastestDet::decode(float *output, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    int num_grid_x = m_stCfg.baseCfg.netWidth / 16;
    int num_grid_y = m_stCfg.baseCfg.netHeight / 16;
    // LOGE("\n");
    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            int   p        = i * num_grid_x + j;
            float box_prob = output[p];
            // LOGE("%.4f ", box_prob);
            if (box_prob < m_stCfg.baseCfg.thresh)
                continue;

            int   category;
            float max_score = 0.0f;
            for (int class_idx = 0; class_idx < m_stCfg.baseCfg.nClasses; class_idx++) {
                float box_cls_score = output[p + (class_idx + 5) * num_grid_x * num_grid_y];
                if (box_cls_score > max_score) {
                    max_score = box_cls_score;
                    category  = class_idx;
                }
            }

            float score = std::pow(max_score, 0.4) * std::pow(box_prob, 0.6);
            if (score < m_stCfg.baseCfg.thresh)
                continue;

            // class loop
            float x_offset = Tanh(output[p + 1 * num_grid_x * num_grid_y]);
            float y_offset = Tanh(output[p + 2 * num_grid_x * num_grid_y]);
            float w        = Sigmoid(output[p + 3 * num_grid_x * num_grid_y]);
            float h        = Sigmoid(output[p + 4 * num_grid_x * num_grid_y]);

            float cx = (j + x_offset) / num_grid_x;
            float cy = (i + y_offset) / num_grid_y;

            // LOGE("x = %.2f, y = %.2f, w = %.2f, h = %.2f, s = %.6f, label = %d \n", x0, y0, w, h, box_prob, class_idx);

            Object obj;
            obj.xmin  = (cx - 0.5 * w) * m_stCfg.baseCfg.netWidth;
            obj.ymin  = (cy - 0.5 * h) * m_stCfg.baseCfg.netHeight;
            obj.xmax  = (cx + 0.5 * w) * m_stCfg.baseCfg.netWidth;
            obj.ymax  = (cy + 0.5 * h) * m_stCfg.baseCfg.netHeight;
            obj.label = category;
            obj.score = score;

            all_bbox_rects.push_back(obj);
            all_bbox_scores.push_back(score);
        }
    }
}
