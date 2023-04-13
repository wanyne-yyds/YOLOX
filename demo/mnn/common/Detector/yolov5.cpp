#include "yolov5.h"
#include <cmath>
int BSJ_AI::YOLOV5::init(const Config &cfg) {
    int nResult = this->baseInit(cfg, "yolov5");
    if (nResult != BSJ_AI_FLAG_SUCCESSFUL) {
        return nResult;
    }
    BSJ_AI_MODEL_FORWARD_ASSERT(m_stCfg.baseCfg.strides.size() != m_stCfg.anchors.size(), "anchors.size() != strides.size()");
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int BSJ_AI::YOLOV5::detect(const ImageData &inputData, std::vector<Object> &vecObjects) {
    vecObjects.clear();

    // gather all box
    std::vector<Object> all_bbox_rects;
    std::vector<float>  all_bbox_scores;

    BSJ_AI::Inference::CallBack func_call_back = [&](int index, std::vector<float *> outputs, std::vector<BSJ_AI::Inference::NCHW> &shapes) {
        if (shapes.size() == 0) {
            LOGE("BSJ_AI::YOLOX::detect  shapes.size() == 0\n");
            return;
        }
        this->decode(outputs[0], m_stCfg.anchors[index], m_stCfg.baseCfg.strides[index], all_bbox_rects, all_bbox_scores);
    };

    int nResult = m_hForward->run(inputData, func_call_back);

    this->postprocess(inputData, vecObjects, all_bbox_rects, all_bbox_scores);

    return BSJ_AI_FLAG_SUCCESSFUL;
}

void BSJ_AI::YOLOV5::decode(float *feat_blob, std::vector<float> anchor, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    int num_anchors = anchor.size() / 2;

    int num_grid_x = m_stCfg.baseCfg.netWidth / stride;
    int num_grid_y = m_stCfg.baseCfg.netHeight / stride;

    int channels_per_box = (4 + 1 + m_stCfg.baseCfg.nClasses);
    int channels         = num_anchors * channels_per_box;

    int step = num_grid_x * num_grid_y * channels_per_box;

    for (int k = 0; k < num_anchors; k++) {
        float anchor_w = anchor[k * 2];     // ¿í
        float anchor_h = anchor[k * 2 + 1]; // ¸ß

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                int   index     = (i * num_grid_x + j) * channels_per_box;
                float box_score = (float)feat_blob[k * step + index + 4]; // obj score
                box_score       = this->Sigmoid(box_score);
                // if (box_score < m_stCfg.baseCfg.thresh) {
                //     continue;
                // }

                int   class_index = 0;
                float class_score = (float)feat_blob[k * step + index + 4 + (0 + 1)];
                for (int q = 1; q < m_stCfg.baseCfg.nClasses; q++) {
                    float score = (float)feat_blob[k * step + index + 4 + (q + 1)];
                    if (score > class_score) {
                        class_index = q;
                        class_score = score;
                    }
                }
                class_score = this->Sigmoid(class_score);
                float score = box_score * class_score;

                if (score < m_stCfg.baseCfg.thresh) {
                    continue;
                }

                float x = (float)feat_blob[k * step + index + 0]; // x center
                float y = (float)feat_blob[k * step + index + 1]; // y center
                float w = (float)feat_blob[k * step + index + 2];
                float h = (float)feat_blob[k * step + index + 3];

                float dx = this->Sigmoid(x);
                float dy = this->Sigmoid(y);
                float dw = this->Sigmoid(w);
                float dh = this->Sigmoid(h);

                float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                float pb_w = std::pow(dw * 2.f, 2) * anchor_w;
                float pb_h = std::pow(dh * 2.f, 2) * anchor_h;

                Object bbox;
                bbox.xmin  = pb_cx - pb_w * 0.5f;
                bbox.ymin  = pb_cy - pb_h * 0.5f;
                bbox.xmax  = pb_cx + pb_w * 0.5f;
                bbox.ymax  = pb_cy + pb_h * 0.5f;
                bbox.label = class_index;
                bbox.score = score;
                all_bbox_scores.push_back(score);
                all_bbox_rects.push_back(bbox);
            }
        }
    }
}
