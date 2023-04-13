#include "rtmdet.h"
#include <cmath>

int BSJ_AI::RTMDET::init(const Config &cfg) {
    return this->baseInit(cfg, "rtmdet");
}

int BSJ_AI::RTMDET::detect(const ImageData &inputData, std::vector<Object> &vecObjects) {
    vecObjects.clear();

    // gather all box
    std::vector<Object> all_bbox_rects;
    std::vector<float> all_bbox_scores;

    BSJ_AI::Inference::CallBack func_call_back = [&](int index, std::vector<float*> outputs, std::vector<BSJ_AI::Inference::NCHW>& shapes) {
        if (shapes.size() == 0 || outputs.size() != 2) {
            LOGE("BSJ_AI::RTMDET::detect  shapes.size() == 0 || outputs.size()[%d] != 3)\n", int(outputs.size()));
            // return BSJ_AI_FLAG_FAILED;
        }
        int i = index % 3;
        // reg, cls
        this->decode(outputs[0], outputs[1], m_stCfg.baseCfg.strides[i], all_bbox_rects, all_bbox_scores);
    };
    int nResult = m_hForward->run(inputData, func_call_back);

    this->postprocess(inputData, vecObjects, all_bbox_rects, all_bbox_scores);

    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::RTMDET::decode(float *preds_reg, float *preds_cls, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    int num_grid_x = m_stCfg.baseCfg.netWidth / stride;
    int num_grid_y = m_stCfg.baseCfg.netHeight / stride;
    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {

            // find label with max score
            int label = -1;
            float score = -1;
            int p = i * num_grid_x + j;

            for (int k = 0; k < m_stCfg.baseCfg.nClasses; k++)
            {
                float s = preds_cls[p + k * num_grid_x * num_grid_y];
                if (s > score)
                {
                    label = k;
                    score = s;
                }
            }

            if (score < m_stCfg.baseCfg.thresh)
                continue;

            float x0 = (j - preds_reg[p + 0 * num_grid_x * num_grid_y]) * stride;
            float y0 = (i - preds_reg[p + 1 * num_grid_x * num_grid_y]) * stride;
            float x1 = (j + preds_reg[p + 2 * num_grid_x * num_grid_y]) * stride;
            float y1 = (i + preds_reg[p + 3 * num_grid_x * num_grid_y]) * stride;

            // LOGE("xmin = %.2f, ymin = %.2f, xmax = %.2f, ymax = %.2f, s = %.6f, label = %d \n", x0, y0, x1, y1, score, label);

            Object obj;
            obj.xmin = x0;
            obj.ymin = y0;
            obj.xmax = x1;
            obj.ymax = y1;
            obj.label = label;
            obj.score = score;

            all_bbox_rects.push_back(obj);
            all_bbox_scores.push_back(score);
        }   
    }
}