#ifndef _DETECTOR_H_
#define _DETECTOR_H_
#include <iostream>
#include <memory>
#include <vector>
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "Inference/Inference.h"

#define BSJ_AI_DETECTOR_VERSION "v1.0.4.a.20230321"

namespace BSJ_AI {
    class Detector {
    public:
        struct Config {
            Inference::Config baseCfg;

            // nms 阈值
            float nms_thresh = 0.45f;

            // 下采样大小
            std::vector<float> strides;
            // 锚框 yolo-anchchor, ssd-min_sizes
            std::vector<std::vector<float>> anchors;
            // 偏移量ssd用到
            std::vector<float> variances;

            // 是否允许检测结果越界
            bool allow_cross_border = false;
        };

        struct Object {
            float xmin  = 0.f;
            float ymin  = 0.f;
            float xmax  = 0.f;
            float ymax  = 0.f;
            float score = 0.f;
            float yaw   = 0.f;
            float pitch = 0.f;
            float roll  = 0.f;
            int   label = 0;

            BSJ_AI::Rect getRect() const {
                BSJ_AI::Rect rect;
                rect.x      = xmin;
                rect.y      = ymin;
                rect.width  = xmax - xmin + 1;
                rect.height = ymax - ymin + 1;
                return rect;
            }
        };

    public:
        Detector();
        ~Detector();
        virtual int init(const Config &cfg) = 0;

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects) = 0;

    protected:
        int baseInit(const Config &cfg, const std::string &funcname);

        void postprocess(const ImageData &inputData, std::vector<Object> &vecObjects, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);

        void qsort_descent_inplace(std::vector<Object> &datas, std::vector<float> &scores);

        void qsort_descent_inplace(std::vector<Object> &datas, std::vector<float> &scores, int left, int right);

        void nms_sorted_bboxes(const std::vector<Object> &bboxes, std::vector<size_t> &picked, float nms_threshold);

        float intersection_area(const Object &a, const Object &b);

        float Sigmoid(float x);

        float Tanh(float x);

        std::shared_ptr<Inference> m_hForward;
        Config                     m_stCfg;
    };
} // namespace BSJ_AI

#endif // _DETECTOR_H_
