#include "detector.h"
#include <cmath>

BSJ_AI::Detector::Detector() {
    m_hForward = std::make_shared<Inference>();
}

BSJ_AI::Detector::~Detector() {
}

int BSJ_AI::Detector::baseInit(const Config &cfg, const std::string &funcname) {
    LOGE("BSJ_AI::Detector::baseInit func %s\n", funcname.c_str());

    m_hForward->init(cfg.baseCfg);
    // ��ֵ
    m_stCfg = cfg;
    return 0;
}

void BSJ_AI::Detector::postprocess(const ImageData &inputData, std::vector<Object> &vecObjects, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores) {
    vecObjects.clear();
    // ����
    this->qsort_descent_inplace(all_bbox_rects, all_bbox_scores);

    // apply nms
    std::vector<size_t> picked;
    this->nms_sorted_bboxes(all_bbox_rects, picked, m_stCfg.nms_thresh);

    int count = picked.size();

    float h_ratio = float(inputData.imgHeight) / m_stCfg.baseCfg.netHeight;
    float w_ratio = float(inputData.imgWidth) / m_stCfg.baseCfg.netWidth;

    vecObjects.resize(count);
    for (int i = 0; i < count; i++) {
        Object obj = all_bbox_rects[picked[i]];

        obj.xmin *= w_ratio;
        obj.ymin *= h_ratio;
        obj.xmax *= w_ratio;
        obj.ymax *= h_ratio;
        if (!m_stCfg.allow_cross_border) {
            obj.xmin = BSJ_MAX(BSJ_MIN(obj.xmin, (float)inputData.imgWidth - 1), 0.f);
            obj.ymin = BSJ_MAX(BSJ_MIN(obj.ymin, (float)inputData.imgHeight - 1), 0.f);
            obj.xmax = BSJ_MAX(BSJ_MIN(obj.xmax, (float)inputData.imgWidth - 1), 0.f);
            obj.ymax = BSJ_MAX(BSJ_MIN(obj.ymax, (float)inputData.imgHeight - 1), 0.f);
        }

        vecObjects[i] = obj;
    }

    // ����
    std::sort(vecObjects.begin(), vecObjects.end(), [](Object a, Object b) {
        return a.score > b.score;
    });
}

void BSJ_AI::Detector::qsort_descent_inplace(std::vector<Object> &datas, std::vector<float> &scores) {
    if (datas.empty() || scores.empty())
        return;
    qsort_descent_inplace(datas, scores, 0, static_cast<int>(scores.size() - 1));
}

void BSJ_AI::Detector::qsort_descent_inplace(std::vector<Object> &datas, std::vector<float> &scores, int left, int right) {
    int   i = left;
    int   j = right;
    float p = scores[(left + right) / 2];

    while (i <= j) {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

void BSJ_AI::Detector::nms_sorted_bboxes(const std::vector<Object> &bboxes, std::vector<size_t> &picked, float nms_threshold) {
    picked.clear();

    const size_t n = bboxes.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++) {
        const Object &r = bboxes[i];

        float width  = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    for (size_t i = 0; i < n; i++) {
        const Object &a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object &b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

float BSJ_AI::Detector::intersection_area(const Object &a, const Object &b) {
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin) {
        // no intersection
        return 0.f;
    }

    float inter_width  = BSJ_MIN(a.xmax, b.xmax) - BSJ_MAX(a.xmin, b.xmin);
    float inter_height = BSJ_MIN(a.ymax, b.ymax) - BSJ_MAX(a.ymin, b.ymin);

    return inter_width * inter_height;
}

float BSJ_AI::Detector::Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float BSJ_AI::Detector::Tanh(float x) {
    return 2.0f / (1.0f + std::exp(-2 * x)) - 1;
}
