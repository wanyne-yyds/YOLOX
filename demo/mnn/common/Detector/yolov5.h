#ifndef _YOLOV5_H_
#define _YOLOV5_H_

#include "detector.h"

namespace BSJ_AI {
    class YOLOV5 : public Detector {
    public:
        virtual int init(const Config &cfg);

        virtual int detect(const ImageData &inputData, std::vector<Object> &vecObjects);

    private:
        void decode(float *feat_blob, std::vector<float> anchor, int stride, std::vector<Object> &all_bbox_rects, std::vector<float> &all_bbox_scores);
    };
} // namespace BSJ_AI

#endif // !_YOLOV5_H_
