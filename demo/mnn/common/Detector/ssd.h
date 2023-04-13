#ifndef _SSD_H_
#define _SSD_H_

#include "detector.h"

namespace BSJ_AI {
    class SSD : public Detector {
    public:
        virtual int init(const Config &cfg);

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects);

    private:
        void PriorBox();
        void decode(float *score_blob, float *bbox_blob, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);

        std::vector<BSJ_AI::Rect2f> m_vDefaultBoxs;
    };
} // namespace BSJ_AI
#endif // !_SSD_H_
