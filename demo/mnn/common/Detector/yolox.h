#ifndef _YOLOX_H_
#define _YOLOX_H_

#include "detector.h"

namespace BSJ_AI {
    class YOLOX : public Detector {
    public:
        virtual int init(const Config &cfg);

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects);

    private:
        void decode(float *preds_reg, float *preds_obj, float *preds_cls, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);
        void FaceDecode(float *preds_reg, float *preds_obj, float *preds_cls, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);
    };
} // namespace BSJ_AI

#endif // !_YOLOX_H
