#include <algorithm>
#include <mutex>
#include "BSJ_AI_defines.h"
#include "BSJ_AI_config.h"
#include "Mat/Mat.h"
#include "Store/Logger.h"
#include "Detector/yolox.h"
#include <opencv2/core.hpp>
#include "ImageManager/ImageManager.h"
#include "StatisticalQueue/StatisticalQueue.h"
#include <iostream>

#define DETECTOR_INPUT_HEIGHT (320)
#define DETECTOR_INPUT_WIDTH  (576)

std::shared_ptr<BSJ_AI::Detector> m_hDetector;                // 目标检测
std::shared_ptr<ImageManager>   m_hImageManager;            // 图像管理队列

int bsd_yolox_images()
{
    BSJ_AI::Detector::Config cfg;
    m_hDetector.reset(new BSJ_AI::YOLOX());

    #if defined(USE_MNN)
        cfg.baseCfg.forward_type = BSJ_AI::Inference::InferenceType::FORWARD_MNN; // 检测器
        cfg.baseCfg.model_path = "/home/ckn/Code/YOLOX/demo/mnn/model/yolox.mnn";
        cfg.baseCfg.netHeight    = DETECTOR_INPUT_HEIGHT;
        cfg.baseCfg.netWidth     = DETECTOR_INPUT_WIDTH;
    #elif defined(USE_NCNN)
        cfg.baseCfg.forward_type = BSJ_AI::Inference::InferenceType::FORWARD_NCNN; // 检测器
        cfg.baseCfg.model_path = "/home/ckn/Code/YOLOX/demo/mnn/model/yolox.bin";
        cfg.baseCfg.param_path = "/home/ckn/Code/YOLOX/demo/mnn/model/yolox.param";
        cfg.baseCfg.netHeight    = DETECTOR_INPUT_HEIGHT;
        cfg.baseCfg.netWidth     = DETECTOR_INPUT_WIDTH;
    #endif

    const float means[3] = {0.f, 0.f, 0.f};
    const float normals[3] = {1.f, 1.f, 1.f};
    ::memcpy(cfg.baseCfg.mean, means, sizeof(means));
    ::memcpy(cfg.baseCfg.normal, normals, sizeof(normals));
    cfg.baseCfg.sInpNodes = std::vector<std::string>{"data"};
    cfg.baseCfg.sOupNodes = std::vector<std::vector<std::string>>{
        {"bbox8", "obj8", "cls8",},
        {"bbox16", "obj16", "cls16",},
        {"bbox32", "obj32", "cls32",},
        {"bbox64", "obj64", "cls64",}
    };

    double ratio_w = DETECTOR_INPUT_HEIGHT / 1280.0;
    double ratio_h = DETECTOR_INPUT_WIDTH / 720.0;

    cfg.baseCfg.srcFormat = BSJ_AI::IMAGE_FORMAT::BGR888;

    cfg.baseCfg.strides  = std::vector<float>{8.f, 16.f, 32.f, 64.f}; // 下采样大小
    cfg.baseCfg.nClasses = 4;                                  // 输出类别
    cfg.baseCfg.thresh   = 0.4;
    cfg.baseCfg.nThread  = 1; // 多线程
    cfg.nms_thresh       = 0.45;

    static const char* class_names[] = {
        "person",
        "personD",
        "other",
        "ignore"
    };

    int ret = m_hDetector->init(cfg);
    std::string src_path = "/home/ckn/ssd/Data_trainset/s_BSD/ckn_bsd_cocoformat_1/JPEGImages/val/Yescheck";
    std::vector<std::string> images_path;
    BSJ_AI::searchAllFiles(src_path, images_path, "jpg", true);

    for (int i = 0; i < images_path.size(); i++) {

        BSJ_AI::CV::Mat image = BSJ_AI::CV::imread(images_path[i]);

        BSJ_AI::ImageData data = BSJ_AI::ImageData(image.cols * image.rows * 3, (char *)image.data, 720, 1280, BSJ_AI::IMAGE_FORMAT::BGR888);
        
        std::vector<BSJ_AI::Detector::Object> vecObjects;
        int ret = m_hDetector->detect(data, vecObjects);

        BSJ_AI::Rect rect;               // 目标位置
        float score;                     // 目标分数
        int labelId;                     // 目标类别ID

        cv::Mat frame = cv::Mat(image.rows, image.cols, CV_8UC3, (unsigned char*)image.data);

        for (std::vector<BSJ_AI::Detector::Object>::iterator it = vecObjects.begin(); it != vecObjects.end(); it++) {
            rect.x      = it->xmin;
            rect.y      = it->ymin;
            rect.width  = it->xmax - it->xmin + 1;
            rect.height = it->ymax - it->ymin + 1;
            score       = it->score;
            labelId     = it->label;

            fprintf(stderr, "%d = %.5f at %d %d %d %d\n", labelId, score,
                    rect.x, rect.y, rect.width, rect.height);
            
            cv::rectangle(frame, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[it->label], score * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            int x = rect.x;
            int y = rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > frame.cols)
                x = frame.cols - label_size.width;

            cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);

            cv::putText(frame, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        cv::imwrite("./image_out/" + std::to_string(i) + ".jpg", frame);
        break;
    }
    return 0;
}

int main()
{
    bsd_yolox_images();
    return 0;
}