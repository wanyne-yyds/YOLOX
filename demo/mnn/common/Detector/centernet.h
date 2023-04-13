#ifndef _CENTERNET_H_
#define _CENTERNET_H_

#include "detector.h"

namespace BSJ_AI {
    class CenterNet : public Detector {
    public:
        virtual int init(const Config &cfg);

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects);

    private:
        void decode(float *heatmap, float *scale, float *offset, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);
    };
} // namespace BSJ_AI

#endif
