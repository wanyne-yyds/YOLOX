#include "ncnn/layer.h"
#include "ncnn/net.h"
#include "../model/yolox.bin.h"
#include "../model/yolox.param.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>

#define YOLOX_NMS_THRESH  0.45 // nms threshold
#define YOLOX_TARGET_W_SIZE 576  // target image size after resize, might use 416 for small model
#define YOLOX_TARGET_H_SIZE 320  // target image size after resize, might use 416 for small model
#define YOLOX_CONF_THRESH 0.25 // threshold of bounding box prob

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static int detect_yolox(const cv::Mat& bgr, std::vector<Object>& objects)
{

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    float w = YOLOX_TARGET_W_SIZE;
    float h = YOLOX_TARGET_H_SIZE;

    double ratio_w = w / img_w;
    double ratio_h = h / img_h;
    std::cout << w << std::endl;
    std::cout << img_w << std::endl;
    std::cout << ratio_w << std::endl;
    unsigned char* model = (unsigned char*)(yolox_bin);
	unsigned char* param = (unsigned char*)(yolox_param_bin);
	std::vector<int> inpNodes = std::vector<int>{yolox_param_id::BLOB_data};
	std::vector<int> oupNodes = std::vector<int>
    {
        yolox_param_id::BLOB_bbox8,
        yolox_param_id::BLOB_obj8,
        yolox_param_id::BLOB_cls8,
        yolox_param_id::BLOB_bbox16,
        yolox_param_id::BLOB_obj16,
        yolox_param_id::BLOB_cls16,
        yolox_param_id::BLOB_bbox32,
        yolox_param_id::BLOB_obj32,
        yolox_param_id::BLOB_cls32,
        yolox_param_id::BLOB_bbox64,
        yolox_param_id::BLOB_obj64,
        yolox_param_id::BLOB_cls64,
    };
    std::vector<std::vector<float>>	anchors;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);

    int nThread = 1;
    ncnn::Net yolox;
    yolox.load_param(param);
    yolox.load_model(model);
    ncnn::Extractor ex = yolox.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(nThread);

    ex.input(inpNodes[0], in);
    std::vector<float> strides = std::vector<float>{ 8.f, 16.f, 32.f, 64.f};
    std::vector<Object> proposals;
    // std::vector<GridAndStride> grid_strides;
    // generate_grids_and_stride(YOLOX_TARGET_SIZE, strides, grid_strides);
    std::chrono::steady_clock::time_point btime = std::chrono::steady_clock::now();
    for (int anchor_idx = 0; anchor_idx < strides.size(); anchor_idx++)
    {
		ncnn::Mat reg_blob, obj_blob, cls_blob;
		ex.extract(oupNodes[anchor_idx * 3 + 0], reg_blob);
		ex.extract(oupNodes[anchor_idx * 3 + 1], obj_blob);
		ex.extract(oupNodes[anchor_idx * 3 + 2], cls_blob);

        const int stride = strides[anchor_idx];
        int num_grid_x = reg_blob.w;
        int num_grid_y = reg_blob.h;
        int nClasses = 4;
        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                float box_objectness = obj_blob.channel(0).row(i)[j];
                if (box_objectness < YOLOX_CONF_THRESH)
                    continue;
                for (int class_idx = 0; class_idx < nClasses; class_idx++)
                {
                    float box_cls_score = cls_blob.channel(class_idx).row(i)[j];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob < YOLOX_CONF_THRESH)
                        continue;
                    // class loop
                    float x_center = (reg_blob.channel(0).row(i)[j] + j) * stride;
                    float y_center = (reg_blob.channel(1).row(i)[j] + i) * stride;
                    float w = exp(reg_blob.channel(2).row(i)[j]) * stride;
                    float h = exp(reg_blob.channel(3).row(i)[j]) * stride;
                    float x0 = x_center - w * 0.5f;
                    float y0 = y_center - h * 0.5f;
                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = class_idx;
                    obj.prob = box_prob;
                    proposals.push_back(obj);
                }
            }
        }
    }
    std::chrono::steady_clock::time_point etime = std::chrono::steady_clock::now();
    std::cout << "runSession once: " << std::chrono::duration_cast<std::chrono::milliseconds>(etime - btime).count() << " ms" << std::endl;
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, YOLOX_NMS_THRESH);
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        x0 = x0 / ratio_w;
        y0 = y0 / ratio_h;
        x1 = x1 / ratio_w;
        y1 = y1 / ratio_h;
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person",
        "personD",
        "other",
        "ignore"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite("./temp.jpg", image);
    // cv::imshow("image", image);
    // cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolox(m, objects);
    draw_objects(m, objects);

    return 0;
}