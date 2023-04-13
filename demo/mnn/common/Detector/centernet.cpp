#include "centernet.h"
#include <cmath>

int BSJ_AI::CenterNet::init(const Config &cfg) {
    return this->baseInit(cfg, "centernet");
}

int BSJ_AI::CenterNet::detect(const ImageData &inputData, std::vector<Object> &vecObjects) {
    vecObjects.clear();

    // gather all box
    std::vector<Object> all_bbox_rects;
    std::vector<float> all_bbox_scores;

    BSJ_AI::Inference::CallBack func_call_back = [&](int index, std::vector<float*> outputs, std::vector<BSJ_AI::Inference::NCHW>& shapes) {
        if (shapes.size() == 0) {
            return;
        }
        // heatmap, scale, offset
        this->decode(outputs[0], outputs[1], outputs[2], all_bbox_rects, all_bbox_scores);
    };

    int nResult = m_hForward->run(inputData, func_call_back);
    this->postprocess(inputData, vecObjects, all_bbox_rects, all_bbox_scores);

    return 0;
}

void BSJ_AI::CenterNet::decode(float *heatmap, float *scale, float *offset, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    int stride = 4;
    int fea_h = m_stCfg.baseCfg.netHeight / stride;
    int fea_w = m_stCfg.baseCfg.netWidth / stride;
    int spacial_size = fea_w * fea_h;

    std::vector<int> ids;
    // 先取分数
    for (int i = 0; i < fea_h; i++) {
        for (int j = 0; j < fea_w; j++) {
            if (heatmap[i * fea_w + j] > m_stCfg.baseCfg.thresh) {
                ids.push_back(i);
                ids.push_back(j);
            }
        }
    }

    for (int i = 0; i < ids.size() / 2; i++) {
        int id_h = ids[2 * i];
        int id_w = ids[2 * i + 1];
        int index = id_h * fea_w + id_w;

        float h = std::exp(scale[index + spacial_size]) * stride;
        float w = std::exp(scale[index]) * stride;
        float o0 = offset[index + spacial_size];
        float o1 = offset[index];

        // std::cout << h << " " << w << " " << o0 << " " << o1 << std::endl;
        float x1 = (float)BSJ_MAX(0., (id_w + o1) * stride - w / 2);
        float y1 = (float)BSJ_MAX(0., (id_h + o0) * stride - h / 2);
        float x2 = 0, y2 = 0;
        x1 = BSJ_MIN((float)m_stCfg.baseCfg.netWidth, x1);
        y1 = BSJ_MIN((float)m_stCfg.baseCfg.netHeight, y1);
        x2 = BSJ_MIN((float)m_stCfg.baseCfg.netWidth, x1 + w);
        y2 = BSJ_MIN((float)m_stCfg.baseCfg.netHeight, y1 + h);

        // std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

        BSJ_AI::Detector::Object obj;
        obj.xmin = x1;
        obj.ymin = y1;
        obj.xmax = x2;
        obj.ymax = y2;
        obj.score = heatmap[index];
        obj.label = 0;

        all_bbox_rects.push_back(obj);
        all_bbox_scores.push_back(obj.score);
    }
}
