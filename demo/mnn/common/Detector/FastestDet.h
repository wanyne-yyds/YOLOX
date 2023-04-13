#ifndef _FASTEST_DET_H_
#define _FASTEST_DET_H_

#include "detector.h"

namespace BSJ_AI {
    class FastestDet : public Detector {
    public:
        virtual int init(const Config &cfg);

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects);

    private:
        void decode(float *output, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);
    };
} // namespace BSJ_AI

#endif
