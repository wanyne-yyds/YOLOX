#ifndef _RTMDET_H_
#define _RTMDET_H_

#include "detector.h"

namespace BSJ_AI {
    class RTMDET : public Detector {
    public:
        virtual int init(const Config &cfg);

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects);

    private:
        void decode(float *preds_reg, float *preds_cls, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);
    };
} // namespace BSJ_AI

#endif // !_RTMDET_H