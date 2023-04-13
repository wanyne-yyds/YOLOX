#include "Mat.h"
#include "BSJ_AI_neon.h"
#include "exif/exif.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace BSJ_AI {
namespace CV {

Mat imread(const std::string &filename) {
    int width, height, channel;
    unsigned char *image = stbi_load(filename.c_str(), &width, &height, &channel, 0);
    if (nullptr == image) {
        LOGE("BSJ_AI::CV::imread stbi_load err: image is null\n");
        return Mat();
    }

    Mat m(height, width, channel <= 3 ? channel : 3);
    if (channel <= 3) {
        memcpy(m.data, image, width * height * channel * sizeof(unsigned char));
    } else {
        unsigned char *ptr = m.data;
        unsigned char *rbga = image;
        for (int i = 0; i < height * width; i++) {
            for (int c = 0; c < 3; c++) {
                ptr[c] = rbga[c];
            }
            ptr += 3;
            rbga += channel;
        }
    }

    stbi_image_free(image);
    // rbg 转 bgr
    if (channel == 3) {
        unsigned char *p = m.data;
        for (int i = 0; i < height * width; i++) {
            std::swap(p[0], p[2]);
            p += 3;
        }
    }

    int orientation = 0;
    BSJ_AI::ExifReader reader(filename);

    if (reader.parse()) {
        orientation = reader.getTag(BSJ_AI::ExifTagName::ORIENTATION).field_u16;
    }


    Size sz;
    // 旋转矩阵 + 平移
    BSJ_AI::CV::Mat2f H;
    
    // exif
    // https://blog.csdn.net/WuLex/article/details/107930360
    Mat matImage;
    switch (orientation) {
    //case BSJ_AI::IMAGE_ORIENTATION_TL:
    //case BSJ_AI::IMAGE_ORIENTATION_TR:
    case BSJ_AI::IMAGE_ORIENTATION_BR:
    case BSJ_AI::IMAGE_ORIENTATION_BL:
        sz = Size(m.cols, m.rows);
        H = getRotationMatrix2D(BSJ_AI::Point(sz.width / 2, sz.height / 2), 180, 1.f);
        warpPerspective(m, matImage, H, sz);
        break;
    case BSJ_AI::IMAGE_ORIENTATION_LT:
    case BSJ_AI::IMAGE_ORIENTATION_RT:
        sz = Size(m.rows, m.cols);
        // 旋转
        H = getRotationMatrix2D(BSJ_AI::Point(sz.width / 2, sz.height / 2), -90, 1.f);
        // 平移
        H.data[2] = sz.width;
        H.data[5] = 0;
        warpPerspective(m, matImage, H, sz);
        break;
    case BSJ_AI::IMAGE_ORIENTATION_RB:
    case BSJ_AI::IMAGE_ORIENTATION_LB:
        sz = Size(m.rows, m.cols);
        // 旋转
        H = getRotationMatrix2D(BSJ_AI::Point(sz.width / 2, sz.height / 2), 90, 1.f);
        // 平移
        H.data[2] = 0;
        H.data[5] = sz.height;
        warpPerspective(m, matImage, H, sz);
        break;
    case BSJ_AI::IMAGE_ORIENTATION_TL:
    case BSJ_AI::IMAGE_ORIENTATION_TR:
    default:
        matImage = m;
        break;
    }
    return matImage;
}

bool imwrite(const std::string &filename, const Mat &image) {
    if (image.empty()) {
        LOGE("BSJ_AI::CV::imwrite err: image is empty\n");
        return false;
    }

    // 获取文件后缀名
    std::string extname = filename.substr(filename.rfind('.') + 1, -1);
    std::transform(extname.begin(), extname.end(), extname.begin(), ::tolower);

    if (extname == "jpg" || extname == "jpeg") {
        return stbi_write_jpg(filename.c_str(), image.cols, image.rows, image.channels, image.data, 95);
    } else if (extname == "png") {
        return stbi_write_png(filename.c_str(), image.cols, image.rows, image.channels, image.data, 0);
    } else if (extname == "bmp") {
        return stbi_write_bmp(filename.c_str(), image.cols, image.rows, image.channels, image.data);
    }

    return false;
}

Mat copyMakeBorder(Mat &image, int top, int bottom, int left, int right, int value) {
    int height = image.rows + top + bottom;
    int width = image.cols + left + right;
    int channel = image.channels;
    Mat dst(height, width, channel);

    int left_plane = left * channel;
    int right_plane = right * channel;

    int src_stride = image.cols * channel;
    int dst_stride = dst.cols * channel;

    unsigned char *src_ptr = (unsigned char *)image.data;
    unsigned char *dst_ptr = (unsigned char *)dst.data;

    int top_plane = top * dst_stride;
    fill(dst_ptr, value, top_plane);
    dst_ptr += top_plane;

    for (int row = 0; row < image.rows; row++) {
        fill(dst_ptr, value, left_plane);
        dst_ptr += left_plane;
        memcpy(dst_ptr, src_ptr, src_stride);
        src_ptr += src_stride;
        dst_ptr += src_stride;
        fill(dst_ptr, value, right_plane);
        dst_ptr += right_plane;
    }
    int bottom_plane = bottom * dst_stride;
    fill(dst_ptr, value, bottom_plane);

    return dst;
}

}
} // namespace BSJ_AI::CV