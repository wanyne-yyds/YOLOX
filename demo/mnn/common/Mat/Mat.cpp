#include "Mat.h"

namespace BSJ_AI {
namespace CV {
void from_yuv_roi_yuv420sp(const Mat& srcImage, Mat& dstImage, Rect roi) {
    int x = (roi.x >> 1 << 1);
    int y = (roi.y >> 1 << 1);
    int w = (roi.width >> 1 << 1);
    int h = (roi.height >> 1 << 1);

    dstImage = Mat(h + (h >> 1), w, 1);
    unsigned char* dst_pixel = (unsigned char*)dstImage.data;
    //Mat tmp;
    //cvtColor(srcImage, tmp, ColorConversionType::COLOR_CONVERT_NV12TORGB);

    /*
    * Y Y
    * Y Y
    * U V
    */

    // copy y
    for (int row = y; row < (y + h); row++) {
        unsigned char* src_pixel = (unsigned char*)srcImage.data + row * srcImage.cols + x;
        memcpy(dst_pixel, src_pixel, w); // memcpy ±È½Ï¿ì
        dst_pixel += w;
    }


    // copy uv
    int start = srcImage.rows * 2.f / 3.f + (y >> 1);
    int end = start + (h >> 1);
    for (int row = start; row < end; row++) {
        unsigned char* src_pixel = (unsigned char*)srcImage.data + row * srcImage.cols + x;
        memcpy(dst_pixel, src_pixel, w);
        dst_pixel += w;
    }

}

void from_yuv_resize_yuv420sp(uint8_t* __restrict src, uint8_t* __restrict dst, Size src_size, Size dst_size) {
    register int sw = src_size.width; // register keyword is for local var to accelorate
    register int sh = src_size.height;
    register int dw = dst_size.width;
    register int dh = dst_size.height;
    register int y, x;
    unsigned long int srcy, srcx, src_index, dst_index;
    unsigned long int xrIntFloat_16 = (sw << 16) / dw + 1; // better than float division
    unsigned long int yrIntFloat_16 = (sh << 16) / dh + 1;

    uint8_t* dst_uv = dst + dh * dw; // memory start pointer of dest uv
    uint8_t* src_uv = src + sh * sw; // memory start pointer of source uv
    uint8_t* dst_uv_yScanline;
    uint8_t* src_uv_yScanline;
    uint8_t* dst_y_slice = dst; // memory start pointer of dest y
    uint8_t* src_y_slice;
    uint8_t* sp;
    uint8_t* dp;

    for (y = 0; y < (dh & ~7); ++y) {
        srcy = (y * yrIntFloat_16) >> 16;
        src_y_slice = src + srcy * sw;

        if ((y & 1) == 0) {
            dst_uv_yScanline = dst_uv + (y / 2) * dw;
            src_uv_yScanline = src_uv + (srcy / 2) * sw;
        }

        for (x = 0; x < (dw & ~7); ++x) {
            srcx = (x * xrIntFloat_16) >> 16;
            dst_y_slice[x] = src_y_slice[srcx];

            if ((y & 1) == 0) // y is even
            {
                if ((x & 1) == 0) // x is even
                {
                    src_index = (srcx / 2) * 2;
                    sp = dst_uv_yScanline + x;
                    dp = src_uv_yScanline + src_index;
                    *sp = *dp;
                    ++sp;
                    ++dp;
                    *sp = *dp;
                }
            }
        }
        dst_y_slice += dw;
    }
}

int fromYuvRoi(const ImageData& inputData, Rect roi, ImageData& dst) {
    if (inputData.data == NULL || inputData.imgWidth * inputData.imgHeight == 0 || inputData.dataLength == 0) {
        LOGE("BSJ_AI::CV::fromYuvRoi inputData.format = %d, inputData.dataLength = %d,  inputData.imgHeight = %d, inputData.imgWidth = %d, inputData.data = %p\n",
            inputData.format, inputData.dataLength, inputData.imgHeight, inputData.imgWidth, inputData.data);
        return -1;
    }

    switch (inputData.format) {
    case IMAGE_FORMAT::NV12:
    case IMAGE_FORMAT::NV21:
    {
        Mat srcImage = Mat(inputData.imgHeight + (inputData.imgHeight >> 1), inputData.imgWidth, 1, (unsigned char*)inputData.data);
        Mat dstImage;
        from_yuv_roi_yuv420sp(srcImage, dstImage, roi);
        Size dSize = Size(dstImage.cols, dstImage.rows * 2.f / 3.f);

        int outLength = dstImage.rows * dstImage.cols * dstImage.channels;
        if (outLength > dst.dataLength) {
            LOGE("BSJ_AI::CV::fromYuvRoi err dst buff size [%d] < roi size [%d]\n", dst.dataLength, outLength);
            return BSJ_AI_FLAG_BAD_PARAMETER;
        }
        memcpy(dst.data, dstImage.data, outLength);
        dst.imgHeight = dSize.height;
        dst.imgWidth = dSize.width;
        dst.dataLength = outLength;
        dst.format = inputData.format;
        break;
    }
    default:
        LOGE("BSJ_AI::CV::fromYuvRoi err: No Supported format = %d!\n", inputData.format);
        break;
    }

    return 0;
}

int fromYuvRoiResize(const ImageData& inputData, Rect roi, Size dstImageSize, ImageData& dst, int align) {
    if (inputData.data == NULL || inputData.imgWidth * inputData.imgHeight == 0 || inputData.dataLength == 0) {
        LOGE("BSJ_AI::CV::fromYuvRoiResize inputData.format = %d, inputData.dataLength = %d,  inputData.imgHeight = %d, inputData.imgWidth = %d, inputData.data = %p\n",
            inputData.format, inputData.dataLength, inputData.imgHeight, inputData.imgWidth, inputData.data);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (dstImageSize.width % 2 != 0 || dstImageSize.height % 2 != 0) {
        LOGE("BSJ_AI::CV::fromYuvRoiResize dstImageSize.width[%d] %% 2 != 0, dstImgHeight[%d] %% 2 != 0\n", dstImageSize.width, dstImageSize.height);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (roi.area() == 0 || roi.x < 0 || roi.y < 0 || (roi.x + roi.width) > inputData.imgWidth || (roi.y + roi.height) > inputData.imgHeight) {
        LOGE("BSJ_AI::CV::CV::fromYuvRoiResize roi[%d, %d, %d, %d] out of image size[0, 0, %d, %d]\n", roi.x, roi.y, roi.width, roi.height, inputData.imgWidth, inputData.imgHeight);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (align % 4 != 0) {
        LOGE("BSJ_AI::CV::fromYuvRoiResize align[%d] %% 4 != 0\n", align);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    switch (inputData.format) {
    case IMAGE_FORMAT::NV12: {
        //// 1. crop roi
        Mat srcImage = Mat(inputData.imgHeight + (inputData.imgHeight>>1), inputData.imgWidth, 1, (unsigned char*)inputData.data);
        Mat dstImage;
        from_yuv_roi_yuv420sp(srcImage, dstImage, roi);
        Size dSize = Size(dstImage.cols, dstImage.rows * 2.f / 3.f);
        int outLength = dstImage.rows * dstImage.cols * dstImage.channels;
        
        float wScale = dstImageSize.width * 1.0 / inputData.imgWidth;
        float hScale = dstImageSize.height * 1.0 / inputData.imgHeight;

        // align
        Size dstSize = Size(0, 0);
        dstSize.width   = (int(dSize.width * wScale) / align) * align;
        dstSize.height  = (int(dSize.height * hScale) / align) * align;
        dstSize.width   = (dstSize.width == 0 ? 32 : dstSize.width);
        dstSize.height  = (dstSize.height == 0 ? 32 : dstSize.height);

        // 2. verify whether the buffer is satisfied
        int dstBufLen   = dstSize.width * (dstSize.height + (dstSize.height >> 1));
        if (dstBufLen > dst.dataLength) {
            LOGE("BSJ_AI::CV::fromYuvRoiResize err dst buff size [%d] < roi size [%d] or dst buff size == 0\n", dst.dataLength, dstBufLen);
            return BSJ_AI_FLAG_BAD_PARAMETER;
        }

        // 3. resize roi
        from_yuv_resize_yuv420sp((uint8_t *)dstImage.data, (uint8_t *)dst.data, dSize, dstSize);
        dst.imgHeight   = dstSize.height;
        dst.imgWidth    = dstSize.width;
        dst.dataLength  = dstBufLen;
        break;
    }
    default:
        LOGE("BSJ_AI::CV::fromYuvRoiResize err: No Supported format = %d!\n", inputData.format);
        break;
    }
    return 0;
}

bool cropImage(const Mat &img, const Rect &r, Mat &matCrop, const Size &resize_wh) {
    if (r.width <= 0 || r.height <= 0) {
        return false;
    }

    Rect rect_image(0, 0, img.cols, img.rows);

    if (r == (r & rect_image) && (resize_wh.width > 0 || resize_wh.height > 0)) {
        resize(img(r), matCrop, resize_wh);
    } else if (resize_wh.width > 0 || resize_wh.height > 0) {
        int wofs = r.width / resize_wh.width / 2;
        int hofs = r.height / resize_wh.height / 2;
        std::vector<Point2f> src_pts(3);
        src_pts[0] = Point2f(r.x + wofs, r.y + hofs);
        src_pts[1] = Point2f(r.x + r.width - 1 - wofs, r.y + hofs);
        src_pts[2] = Point2f(r.x + wofs, r.y + r.height - 1 - hofs);

        std::vector<Point2f> dst_pts(3);
        dst_pts[0] = Point2f(0, 0);
        dst_pts[1] = Point2f(resize_wh.width - 1, 0);
        dst_pts[2] = Point2f(0, resize_wh.height - 1);
        //Mat rotateMat = getAffineTransform(src_pts, dst_pts);

        //int flag = cv::INTER_LINEAR;
        //warpAffine(img, matCrop, rotateMat, resize_wh, flag);
    } else {
        Rect rect = r;

        if (matCrop.channels != img.channels || matCrop.rows != rect.height || matCrop.cols != rect.width) {
            matCrop = Mat(rect.height, rect.width, img.channels);
            matCrop.setZeros();
        }

        int dx = BSJ_ABS(BSJ_MIN(0, rect.x));
        if (dx > 0) {
            rect.x = 0;
        }
        rect.width -= dx;
        int dy = BSJ_ABS(BSJ_MIN(0, rect.y));
        if (dy > 0) {
            rect.y = 0;
        }
        rect.height -= dy;
        int dw = BSJ_ABS(BSJ_MIN(0, img.cols - (rect.x + rect.width)));
        rect.width -= dw;
        int dh = BSJ_ABS(BSJ_MIN(0, img.rows - (rect.y + rect.height)));
        rect.height -= dh;
        if (rect.width > 0 && rect.height > 0) {
            int step1 = img.cols * img.channels;
            int step2 = matCrop.cols * matCrop.channels;
            
            for (int i = dy; i < dy + rect.height; i++) {
                unsigned char *src = img.data + (rect.y + i - dy) * step1 + rect.x * img.channels;
                unsigned char *dst = matCrop.data + i * step2 + dx * matCrop.channels;
                memcpy(dst, src, rect.width * img.channels);
            }

        }
    }

    return true;
}
}
} // namespace BSJ_AI::CV
