#include "Mat.h"
#include <memory>
#include <cmath>
#include <climits>

namespace BSJ_AI {
namespace CV {
ResizeNEAREST::ResizeNEAREST(const IMAGE_FORMAT &format, int srcH, int srcW, int dstH, int dstW) {
    m_eFormat = format;
    this->init(srcH, srcW, dstH, dstW);
    m_nDstH = dstH;
    m_nDstW = dstW;
}

ResizeNEAREST::~ResizeNEAREST() {
    m_arrX.clear();
    m_arrY.clear();
}

void ResizeNEAREST::init(int srcH, int srcW, int dstH, int dstW) {
    if (srcH <= 0 || srcW <= 0 || dstH <= 0 || dstW <= 0) {
        LOGE("BSJ_AI::ResizeNEAREST::init err srcH = %d,  srcW = %d, dstH = %d, dstW = %d\n", srcH, srcW, dstH, dstW);
        return;
    }

    // 计算透视变换矩阵
    /* https://github.com/2892211452/MDimg/blob/master/copyMD/%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86.md
     * X     a11 a12 a13     x
     * Y =   a21 a22 a23 *   y
     * Z     a31 a32 a33     1
     * 进行投影变换
     * X' = X / Z = (a11x + a12y + a13) / (a31x + a32y + a33)
     * Y' = Y / Z = (a21x + a22y + a23) / (a31x + a32y + a33)
     * Z' = Z / Z = 1
     * 令 a33 = 1. 则有
     * a11x + a12y + a13 - a31xX' - a32yX' = X'
     * a21x + a22y + a23 - a31xY' - a32yY' = Y'
     * 1. 代入 (0, 0), (0, 0) 求出 a13, a23 -> m[2], m[5]
     * 2. 代入 (2560, 0), (1280, 0) 求出 a21-> m[3], (srcW)*a11 - (srcW * dstW)*a31 = dstW
     * 3. 代入 (0, 1440), (0, 720) 求出 a12 -> m[1], (srcH)*a22 - (srcH * dstH)*a32 = dstH
     * 4. 代入 (2560, 1440), (1280, 720) 求出 a32 a31 a22
     *   (srcW)*a11 - (srcW * dstW)*a31 - (srcH * dstW) * a32 = dstW
     *   (srcH)*a22 - (srcW * dstH)*a31 - (srcH * dstH) * a32 = dstH
     * opencv使用 cv::Mat H = cv::getPerspectiveTransform(srcPts, dstPts);
     */
    std::vector<float> m;
    m.resize(9, 0.f);

    // a33
    m[8] = 1.f;

    m[2] = 0.f;
    m[5] = 0.f;

    m[3] = 0.f / srcW;

    m[1] = 0.f / srcH;

    m[7] = float(dstW - dstW) / (srcH * dstW); // a32;
    m[6] = float(dstH - dstH) / (srcH * dstW); // a31

    m[0] = float(dstW) / srcW; // a11
    m[4] = float(dstH) / srcH; // a22

    /* 求逆矩阵 1.待定系数法， 2.伴随矩阵法， 3.初等变换法。
     *  伴随矩阵法
     *  A^(-1) = A^(*) / |A|
     *  求矩阵的逆
     *  a11 a12 a13
     *  a21 a22 a23
     *  a31 a32 a33
     *  行列式
     *  | A | = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
     *  伴随矩阵
     *  A11 = a22 * a33 - a23 * a32
     *  A12 = a21 * a33 - a23 * a31
     *  ...
     *  https://www.bilibili.com/read/cv2920478
     *  opencv 使用mat.inv()
     */

    std::vector<float> m_inv;
    m_inv.resize(9, 0.f);

    float determinant = m[0] * (m[4] * m[8] - m[5] * m[7])
                        - m[1] * (m[3] * m[8] - m[5] * m[6])
                        + m[2] * (m[3] * m[7] - m[4] * m[6]);

    m_inv[0] = (m[4] * m[8] - m[5] * m[7]) / determinant;
    m_inv[3] = -(m[3] * m[8] - m[5] * m[6]) / determinant;
    m_inv[6] = (m[3] * m[7] - m[4] * m[6]) / determinant;
    m_inv[1] = -(m[1] * m[8] - m[2] * m[7]) / determinant;
    m_inv[4] = (m[0] * m[8] - m[2] * m[6]) / determinant;
    m_inv[7] = -(m[0] * m[7] - m[1] * m[6]) / determinant;
    m_inv[2] = (m[1] * m[5] - m[2] * m[4]) / determinant;
    m_inv[5] = -(m[0] * m[5] - m[2] * m[3]) / determinant;
    m_inv[8] = (m[0] * m[4] - m[1] * m[3]) / determinant;

    /* 获取映射矩阵
     *
     */
    int rows = dstH;
    int cols = dstW;
    if (m_eFormat == IMAGE_FORMAT::NV12 || m_eFormat == IMAGE_FORMAT::NV21) {
        rows = dstH + (dstH >> 1);
    }

    m_arrX.resize(rows * cols);
    m_arrY.resize(rows * cols);

    float x3 = m_inv[2] * 1;
    float y3 = m_inv[5] * 1;
    float z3 = m_inv[8] * 1;

    //#ifdef USE_NEON
    //    float32x4_t offset = {0, 1, 2, 3};
    //    float32x4_t a11 = vdupq_n_f32(m_inv[0]);
    //    float32x4_t a21 = vdupq_n_f32(m_inv[3]);
    //    float32x4_t a31 = vdupq_n_f32(m_inv[6]);
    //    float32x4_t x33 = vdupq_n_f32(x3);
    //    float32x4_t y33 = vdupq_n_f32(y3);
    //    float32x4_t z33 = vdupq_n_f32(z3);
    //#endif

    // 步长为2，yuv数据和宽高 基本为2的倍数为关系
    for (int row = 0; row < dstH; row += 2) {
        int col = 0;
        float x2 = m_inv[1] * row;
        float y2 = m_inv[4] * row;
        float z2 = m_inv[7] * row;

        //#ifdef USE_NEON
        //        int stride = 4;
        //        float32x4_t x22 = vdupq_n_f32(x2);
        //        float32x4_t y22 = vdupq_n_f32(y2);
        //        float32x4_t z22 = vdupq_n_f32(z2);
        //        for (; col < dstW - stride; col += stride)
        //        {
        //            int32_t* mapX = (int32_t*)&m_arrX[row * dstW + col];
        //            int32_t* mapY = (int32_t*)&m_arrY[row * dstW + col];
        //
        //            //
        //            int32x4_t xi = vld1q_s32(mapX);
        //            int32x4_t yi = vld1q_s32(mapY);
        //
        //            float32x4_t vec = vdupq_n_f32(col);
        //            vec = vaddq_f32(vec, offset);
        //
        //            float32x4_t x;
        //            float32x4_t y;
        //            float32x4_t z;
        //
        //            x = vmulq_f32(vec, a11);
        //            x = vaddq_f32(x, x22);
        //            x = vaddq_f32(x, x33);
        //
        //            y = vmulq_f32(vec, a21);
        //            y = vaddq_f32(y, y22);
        //            y = vaddq_f32(y, y33);
        //
        //            z = vmulq_f32(vec, a31);
        //            z = vaddq_f32(z, z22);
        //            z = vaddq_f32(z, z33);
        //
        //            float32x4_t zii = vrecpeq_f32(z); // vrecpeq_f32倒数
        //            float32x4_t xii = vmulq_f32(x, zii);
        //            float32x4_t yii = vmulq_f32(y, zii);
        //
        //            xi = vcvtnq_s32_f32(xii);
        //            yi = vcvtnq_s32_f32(yii);
        //
        //            vst1q_s32(mapX, xi);
        //            vst1q_s32(mapY, yi);
        //
        //        }
        //#endif // USE_NEON

        for (; col < dstW; col += 2) {
            // nv12的y通道
            for (int i = 0; i < 2; i++) {
                float row_x2 = x2;
                float row_y2 = y2;
                float row_z2 = z2;

                // 加法比乘法快
                if (i) {
                    row_x2 += m_inv[1];
                    row_y2 += m_inv[4];
                    row_z2 += m_inv[7];
                }

                for (int j = 0; j < 2; j++) {
                    float x = m_inv[0] * (col + j) + row_x2 + x3;
                    float y = m_inv[3] * (col + j) + row_y2 + y3;
                    float z = m_inv[6] * (col + j) + row_z2 + z3;

                    int xi = BSJ_BETWEEN(x / z, 0, srcW - 1);
                    int yi = BSJ_BETWEEN(y / z, 0, srcH - 1);

                    int index = (row + i) * dstW + (col + j);
                    m_arrX[index] = xi;
                    m_arrY[index] = yi;
                }
            }

            if (m_eFormat == IMAGE_FORMAT::NV12 || m_eFormat == IMAGE_FORMAT::NV21) {
                float uvx2 = m_inv[1] * (dstH + (row >> 1));
                float uvy2 = m_inv[4] * (dstH + (row >> 1));
                float uvz2 = m_inv[7] * (dstH + (row >> 1));

                // yyyy uv 4个y 共用一个uv
                // U | V
                float xu = m_inv[0] * (col + 0) + uvx2 + x3;
                float yu = m_inv[3] * (col + 0) + uvy2 + y3;
                float zu = m_inv[6] * (col + 0) + uvz2 + z3;

                float xv = m_inv[0] * (col + 1) + uvx2 + x3;
                float yv = m_inv[3] * (col + 1) + uvy2 + y3;
                float zv = m_inv[6] * (col + 1) + uvz2 + z3;

                int xiu = xu / zu;
                int yiu = yu / zu;

                int xiv = xv / zv;
                int yiv = yv / zv;

                yiu = yiu >= dstH ? yiu : dstH;
                yiv = yiv >= dstH ? yiv : dstH;

                // u
                int index = (dstH + (row >> 1)) * dstW + col;
                m_arrX[index] = xiu;
                m_arrY[index] = yiu;

                // v
                index++;
                m_arrX[index] = xiu + 1;
                m_arrY[index] = yiu + 1;

                // LOGE("%d %d %d %d\n", xiu, yiu, xiv, yiv);
            }
        }
    }
}

int ResizeNEAREST::resize(const ImageData &srcData, ImageData &dstData) {
    if (srcData.format != m_eFormat) {
        LOGE("BSJ_AI::ResizeNEAREST::resize error: srcData.format[%d] != m_eFormat[%d]\n", srcData.format, m_eFormat);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (srcData.data == NULL) {
        LOGE("BSJ_AI::ResizeNEAREST::resize error: srcData.data is NULL\n");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (int(m_arrX.size()) == 0 || int(m_arrY.size()) == 0) {
        LOGE("BSJ_AI::ResizeNEAREST::resize error: int(m_arrX.size()) == 0 || int(m_arrY.size()) == 0\n");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    switch (srcData.format) {
    case IMAGE_FORMAT::BGR888:
    case IMAGE_FORMAT::RGB888: {
        if (srcData.dataLength != srcData.imgHeight * srcData.imgWidth * 3) {
            LOGE("BSJ_AI::ResizeNEAREST::resize error: format[%d] srcData.data is NULL srcData.dataLength[%d] != srcData.imgHeight * srcData.imgWidth * 3[%d]\n",
                 m_eFormat, srcData.dataLength, srcData.imgHeight * srcData.imgWidth * 3);
            return BSJ_AI_FLAG_BAD_PARAMETER;
        }

        if (dstData.data != NULL) {
            free(dstData.data);
        }
        dstData.dataLength = m_nDstW * m_nDstH * 3;
        dstData.data = (char *)calloc(dstData.dataLength * sizeof(char), 1);
        dstData.format = srcData.format;
        dstData.imgHeight = m_nDstH;
        dstData.imgWidth = m_nDstW;
        break;
    }
    case IMAGE_FORMAT::GRAY: {
        if (srcData.dataLength != srcData.imgHeight * srcData.imgWidth) {
            LOGE("BSJ_AI::ResizeNEAREST::resize error: format[%d] srcData.data is NULL srcData.dataLength[%d] != srcData.imgHeight * srcData.imgWidth[%d]\n",
                 m_eFormat, srcData.dataLength, srcData.imgHeight * srcData.imgWidth);
            return BSJ_AI_FLAG_BAD_PARAMETER;
        }

        if (dstData.data != NULL) {
            free(dstData.data);
        }
        dstData.dataLength = m_nDstW * m_nDstH;
        dstData.data = (char *)calloc(dstData.dataLength * sizeof(char), 1);
        dstData.format = srcData.format;
        dstData.imgHeight = m_nDstH;
        dstData.imgWidth = m_nDstW;
        break;
    }
    case IMAGE_FORMAT::NV12:
    case IMAGE_FORMAT::NV21: {
        if (srcData.dataLength != srcData.imgHeight * srcData.imgWidth * 1.5) {
            LOGE("BSJ_AI::ResizeNEAREST::resize error: format[%d] srcData.data is NULL srcData.dataLength[%d] != srcData.imgHeight * srcData.imgWidth * 1.5[%d]\n",
                 m_eFormat, srcData.dataLength, int(srcData.imgHeight * srcData.imgWidth * 1.5f));
            return BSJ_AI_FLAG_BAD_PARAMETER;
        }

        if (dstData.data != NULL) {
            free(dstData.data);
        }
        dstData.dataLength = m_nDstW * m_nDstH * 1.5;
        dstData.data = (char *)calloc(dstData.dataLength * sizeof(char), 1);
        dstData.format = srcData.format;
        dstData.imgHeight = m_nDstH;
        dstData.imgWidth = m_nDstW;
        break;
    }
    default:
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    int index = 0;
    for (int row = 0; row < m_nDstH; row += 2) {
        for (int col = 0; col < m_nDstW; col += 2) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    index = (row + i) * m_nDstW + (col + j);
                    int xi = m_arrX[index];
                    int yi = m_arrY[index];

                    if (xi >= 0 && xi < srcData.imgWidth && yi >= 0 && yi < srcData.imgHeight) {
                        int dIndex = 0;
                        int sIndex = 0;
                        if (m_eFormat == IMAGE_FORMAT::BGR888 || m_eFormat == IMAGE_FORMAT::RGB888) {
                            dIndex = (row + i) * m_nDstW * 3 + (col + j) * 3;
                            sIndex = yi * srcData.imgWidth * 3 + xi * 3;
                            dstData.data[dIndex + 0] = srcData.data[sIndex + 0];
                            dstData.data[dIndex + 1] = srcData.data[sIndex + 1];
                            dstData.data[dIndex + 2] = srcData.data[sIndex + 2];
                        } else if (m_eFormat == IMAGE_FORMAT::GRAY || m_eFormat == IMAGE_FORMAT::NV12 || m_eFormat == IMAGE_FORMAT::NV21) {
                            dIndex = (row + i) * m_nDstW + (col + j);
                            sIndex = yi * srcData.imgWidth + xi;
                            dstData.data[dIndex] = srcData.data[sIndex];
                        }
                    }
                }
            }

            if (m_eFormat == IMAGE_FORMAT::NV12 || m_eFormat == IMAGE_FORMAT::NV21) {
                int dIndex = 0;
                int sIndex = 0;

                // u
                index = (m_nDstH + (row >> 1)) * m_nDstW + col;
                int xiu = m_arrX[index];
                int yiu = m_arrY[index];

                // v
                index++;
                int xiv = m_arrX[index];
                int yiv = m_arrY[index];

                dIndex = (m_nDstH + (row >> 1)) * m_nDstW + col;
                // u
                dstData.data[dIndex + 0] = srcData.data[yiu * srcData.imgWidth + xiu];
                // v
                dstData.data[dIndex + 1] = srcData.data[yiv * srcData.imgWidth + xiv];
            }
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

#ifdef USE_NEON

#define MAKE_LOAD(n)                                                 \
    _sx = vld1q_s32(xofs_p);                                         \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0), _S0, 0);     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1), _S0, 1);     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2), _S0, 2);     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3), _S0, 3);     \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0) + n, _S1, 0); \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1) + n, _S1, 1); \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2) + n, _S1, 2); \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3) + n, _S1, 3); \
    _sx = vld1q_s32(xofs_p + 4);                                     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0), _S0, 4);     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1), _S0, 5);     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2), _S0, 6);     \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3), _S0, 7);     \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0) + n, _S1, 4); \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1) + n, _S1, 5); \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2) + n, _S1, 6); \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3) + n, _S1, 7); \
    uint8x8_t _mask = vld1_u8(ialpha_p);

#define LOAD_C1() MAKE_LOAD(1)
#define LOAD_C2() MAKE_LOAD(2)
#define LOAD_C3() MAKE_LOAD(3)
#define LOAD_C4() MAKE_LOAD(4)

#endif // USE_NEON

void ResizeNearestImpl(const Mat &src, Mat &dst, const Size &dsize, double &scale_x, double &scale_y) {
    int channel = src.channels;
    int *buf = new int[dsize.width + dsize.height + dsize.width + dsize.height];

    int *xofs = buf;               // new int[w];
    int *yofs = buf + dsize.width; // new int[h];

    unsigned char *ialpha = (unsigned char *)(buf + dsize.width + dsize.height);              // new short[w * 2];
    unsigned char *ibeta = (unsigned char *)(buf + dsize.width + dsize.height + dsize.width); // new short[h * 2];

    int src_stride = src.cols * channel;
    int dst_stride = dsize.width * channel;

    float fx;
    float fy;
    int sx;
    int sy;

    // x
    for (int dx = 0; dx < dsize.width; dx++) {
        fx = (dx + 0.5) * scale_x - 0.5;
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src.cols - 1) {
            sx = src.cols - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * channel;

        ialpha[dx] = (fx <= 0.5) ? -1 : 0;
    }

    for (int dy = 0; dy < dsize.height; dy++) {
        fy = (dy + 0.5) * scale_y - 0.5;
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src.rows - 1) {
            sy = src.rows - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        ibeta[dy] = (fy <= 0.5) ? -1 : 0;
    }

    for (int dy = 0; dy < dsize.height; dy++) {
        sy = (ibeta[dy] == 0) ? yofs[dy] + 1 : yofs[dy];
        const unsigned char *Sp = (unsigned char *)src.data + src_stride * sy;
        unsigned char *Dp = (unsigned char *)dst.data + dst_stride * dy;

        int dx = 0;

#ifdef USE_NEON
        if (channel == 1) {
            int32x4_t _sx = int32x4_t();
            uint8x8_t _S0 = uint8x8_t();
            uint8x8_t _S1 = uint8x8_t();
            int *xofs_p = xofs;
            uint8_t *ialpha_p = ialpha;
            uint8_t *Dp_p = Dp;
            int simd_loop = 0;
            for (int i = 0; i < dsize.width - 7; i += 8) {
                LOAD_C1();

                vst1_u8(Dp_p, vbsl_u8(_mask, _S0, _S1));

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8;
                ++simd_loop;
            }
            dx += simd_loop * 8;
        } else if (channel == 2) {
            int32x4_t _sx = int32x4_t();
            uint8x8x2_t _S0 = uint8x8x2_t();
            uint8x8x2_t _S1 = uint8x8x2_t();
            uint8x8x2_t _S2 = uint8x8x2_t();
            int *xofs_p = xofs;
            uint8_t *ialpha_p = ialpha;
            uint8_t *Dp_p = Dp;
            int simd_loop = 0;
            for (int i = 0; i < dsize.width - 7; i += 8) {
                LOAD_C2();

                _S2.val[0] = vbsl_u8(_mask, _S0.val[0], _S1.val[0]);
                _S2.val[1] = vbsl_u8(_mask, _S0.val[1], _S1.val[1]);
                vst2_u8(Dp_p, _S2);

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 2;
                ++simd_loop;
            }
            dx += simd_loop * 8;
        } else if (channel == 3) {
            int32x4_t _sx = int32x4_t();
            uint8x8x3_t _S0 = uint8x8x3_t();
            uint8x8x3_t _S1 = uint8x8x3_t();
            uint8x8x3_t _S2 = uint8x8x3_t();
            int *xofs_p = xofs;
            uint8_t *ialpha_p = ialpha;
            uint8_t *Dp_p = Dp;
            int simd_loop = 0;
            for (int i = 0; i < dsize.width - 7; i += 8) {
                LOAD_C3();

                _S2.val[0] = vbsl_u8(_mask, _S0.val[0], _S1.val[0]);
                _S2.val[1] = vbsl_u8(_mask, _S0.val[1], _S1.val[1]);
                _S2.val[2] = vbsl_u8(_mask, _S0.val[2], _S1.val[2]);
                vst3_u8(Dp_p, _S2);

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 3;
                ++simd_loop;
            }
            dx += simd_loop * 8;
        } else if (channel == 4) {
            int32x4_t _sx = int32x4_t();
            uint8x8x4_t _S0 = uint8x8x4_t();
            uint8x8x4_t _S1 = uint8x8x4_t();
            uint8x8x4_t _S2 = uint8x8x4_t();
            int *xofs_p = xofs;
            uint8_t *ialpha_p = ialpha;
            uint8_t *Dp_p = Dp;
            int simd_loop = 0;
            for (int i = 0; i < dsize.width - 7; i += 8) {
                LOAD_C4();

                _S2.val[0] = vbsl_u8(_mask, _S0.val[0], _S1.val[0]);
                _S2.val[1] = vbsl_u8(_mask, _S0.val[1], _S1.val[1]);
                _S2.val[2] = vbsl_u8(_mask, _S0.val[2], _S1.val[2]);
                _S2.val[3] = vbsl_u8(_mask, _S0.val[3], _S1.val[3]);
                vst4_u8(Dp_p, _S2);

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 4;
                ++simd_loop;
            }
            dx += simd_loop * 8;
        }
#endif //  USE_NEON
        for (; dx < dsize.width; dx++) {
            int sx = xofs[dx];
            if (channel == 1) {
                Dp[dx] = (ialpha[dx] == 0) ? Sp[sx + 1] : Sp[sx];
            } else if (channel == 2) {
                Dp[dx * 2] = (ialpha[dx] == 0) ? Sp[sx + 2] : Sp[sx];
                Dp[dx * 2 + 1] = (ialpha[dx] == 0) ? Sp[sx + 3] : Sp[sx + 1];
            } else if (channel == 3) {
                Dp[dx * 3] = (ialpha[dx] == 0) ? Sp[sx + 3] : Sp[sx];
                Dp[dx * 3 + 1] = (ialpha[dx] == 0) ? Sp[sx + 4] : Sp[sx + 1];
                Dp[dx * 3 + 2] = (ialpha[dx] == 0) ? Sp[sx + 5] : Sp[sx + 2];
            } else if (channel == 4) {
                Dp[dx * 4] = (ialpha[dx] == 0) ? Sp[sx + 4] : Sp[sx];
                Dp[dx * 4 + 1] = (ialpha[dx] == 0) ? Sp[sx + 5] : Sp[sx + 1];
                Dp[dx * 4 + 2] = (ialpha[dx] == 0) ? Sp[sx + 6] : Sp[sx + 2];
                Dp[dx * 4 + 3] = (ialpha[dx] == 0) ? Sp[sx + 7] : Sp[sx + 3];
            }
        }
    }
    delete[] buf;
}

void ResizeBilinearImpl(const Mat &src, Mat &dst, const Size &dsize, double &scale_x, double &scale_y) {
    int channel = src.channels;
    int *buf = new int[dsize.width + dsize.height + dsize.width + dsize.height];

    int *xofs = buf;               // new int[w];
    int *yofs = buf + dsize.width; // new int[h];

    short *ialpha = (short *)(buf + dsize.width + dsize.height);              // new short[w * 2];
    short *ibeta = (short *)(buf + dsize.width + dsize.height + dsize.width); // new short[h * 2];

    int src_stride = src.cols * channel;
    int dst_stride = dsize.width * channel;

    float fx;
    float fy;
    int sx;
    int sy;

    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

#define SATURATE_CAST_SHORT1(X) (short)BSJ_MIN(BSJ_MAX((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);
    // x
    for (int dx = 0; dx < dsize.width; dx++) {
        fx = (dx + 0.5) * scale_x - 0.5;
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src.cols - 1) {
            sx = src.cols - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * channel;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT1(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT1(a1);
    }

    for (int dy = 0; dy < dsize.height; dy++) {
        fy = (dy + 0.5) * scale_y - 0.5;
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src.rows - 1) {
            sy = src.rows - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT1(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT1(b1);
    }
#undef SATURATE_CAST_SHORT

    short *rows0 = 0;
    short *rows1 = 0;
    if (channel == 1) {
        rows0 = new short[dsize.width];
        rows1 = new short[dsize.width];
    } else if (channel == 2) {
        rows0 = new short[dsize.width * 2 + 2];
        rows1 = new short[dsize.width * 2 + 2];
    } else if (channel == 3) {
        rows0 = new short[dsize.width * 3 + 1];
        rows1 = new short[dsize.width * 3 + 1];
    } else if (channel == 4) {
        rows0 = new short[dsize.width * 4];
        rows1 = new short[dsize.width * 4];
    }

    int prev_sy = -2;
    for (int dy = 0; dy < dsize.height; dy++) {
        sy = yofs[dy];

        if (sy == prev_sy) {
            // reuse all rows
        } else if (sy == prev_sy + 1) {
            // hresize one row
            short *rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = (unsigned char *)src.data + src_stride * (sy + 1);

            const short *ialphap = ialpha;
            short *rows1p = rows1;
            for (int dx = 0; dx < dsize.width; dx++) {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char *S1p = S1 + sx;
#ifdef USE_NEON
                if (channel == 2) {
                    int16x4_t _a0a1XX = vld1_s16(ialphap);
                    int16x4_t _a0a0a1a1 = vzip_s16(_a0a1XX, _a0a1XX).val[0];
                    uint8x8_t _S1 = uint8x8_t();

                    _S1 = vld1_lane_u8(S1p, _S1, 0);
                    _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                    _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                    _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                    int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                    int16x4_t _S1lowhigh = vget_low_s16(_S116);
                    int32x4_t _S1ma0a1 = vmull_s16(_S1lowhigh, _a0a0a1a1);
                    int32x2_t _rows1low = vadd_s32(vget_low_s32(_S1ma0a1), vget_high_s32(_S1ma0a1));
                    int32x4_t _rows1 = vcombine_s32(_rows1low, vget_high_s32(_S1ma0a1));
                    int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                    vst1_s16(rows1p, _rows1_sr4);
                } else if (channel == 3) {
                    int16x4_t _a0 = vdup_n_s16(a0);
                    int16x4_t _a1 = vdup_n_s16(a1);
                    uint8x8_t _S1 = uint8x8_t();

                    _S1 = vld1_lane_u8(S1p, _S1, 0);
                    _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                    _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                    _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                    _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                    _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                    int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                    int16x4_t _S1low = vget_low_s16(_S116);
                    int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                    int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                    _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                    int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                    vst1_s16(rows1p, _rows1_sr4);
                } else if (channel == 4) {
                    int16x4_t _a0 = vdup_n_s16(a0);
                    int16x4_t _a1 = vdup_n_s16(a1);
                    uint8x8_t _S1 = vld1_u8(S1p);

                    int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                    int16x4_t _S1low = vget_low_s16(_S116);
                    int16x4_t _S1high = vget_high_s16(_S116);
                    int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                    _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                    int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                    vst1_s16(rows1p, _rows1_sr4);
                } else {
                    for (int dc = 0; dc < channel; ++dc) {
                        rows1p[dc] = (S1p[dc] * a0 + S1p[dc + channel] * a1) >> 4;
                    }
                }
#else
                for (int dc = 0; dc < channel; ++dc) {
                    rows1p[dc] = (S1p[dc] * a0 + S1p[dc + channel] * a1) >> 4;
                }
#endif // USE_NEON
                ialphap += 2;
                rows1p += channel;
            }
        } else {
            // hresize two rows
            const unsigned char *S0 = (unsigned char *)src.data + src_stride * (sy);
            const unsigned char *S1 = (unsigned char *)src.data + src_stride * (sy + 1);

            const short *ialphap = ialpha;
            short *rows0p = rows0;
            short *rows1p = rows1;

            for (int dx = 0; dx < dsize.width; dx++) {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char *S0p = S0 + sx;
                const unsigned char *S1p = S1 + sx;

#ifdef USE_NEON
                if (channel == 2) {
                    int16x4_t _a0 = vdup_n_s16(a0);
                    int16x4_t _a1 = vdup_n_s16(a1);
                    uint8x8_t _S0 = uint8x8_t();
                    uint8x8_t _S1 = uint8x8_t();

                    _S0 = vld1_lane_u8(S0p, _S0, 0);
                    _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
                    _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
                    _S0 = vld1_lane_u8(S0p + 3, _S0, 3);

                    _S1 = vld1_lane_u8(S1p, _S1, 0);
                    _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                    _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                    _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                    int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                    int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                    int16x4_t _S0lowhigh = vget_low_s16(_S016);
                    int16x4_t _S1lowhigh = vget_low_s16(_S116);
                    int32x2x2_t _S0S1low_S0S1high = vtrn_s32(vreinterpret_s32_s16(_S0lowhigh), vreinterpret_s32_s16(_S1lowhigh));
                    int32x4_t _rows01 = vmull_s16(vreinterpret_s16_s32(_S0S1low_S0S1high.val[0]), _a0);
                    _rows01 = vmlal_s16(_rows01, vreinterpret_s16_s32(_S0S1low_S0S1high.val[1]), _a1);
                    int16x4_t _rows01_sr4 = vshrn_n_s32(_rows01, 4);
                    int16x4_t _rows1_sr4 = vext_s16(_rows01_sr4, _rows01_sr4, 2);
                    vst1_s16(rows0p, _rows01_sr4);
                    vst1_s16(rows1p, _rows1_sr4);
                } else if (channel == 3) {
                    int16x4_t _a0 = vdup_n_s16(a0);
                    int16x4_t _a1 = vdup_n_s16(a1);
                    uint8x8_t _S0 = uint8x8_t();
                    uint8x8_t _S1 = uint8x8_t();

                    _S0 = vld1_lane_u8(S0p, _S0, 0);
                    _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
                    _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
                    _S0 = vld1_lane_u8(S0p + 3, _S0, 3);
                    _S0 = vld1_lane_u8(S0p + 4, _S0, 4);
                    _S0 = vld1_lane_u8(S0p + 5, _S0, 5);

                    _S1 = vld1_lane_u8(S1p, _S1, 0);
                    _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                    _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                    _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                    _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                    _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                    int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                    int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                    int16x4_t _S0low = vget_low_s16(_S016);
                    int16x4_t _S1low = vget_low_s16(_S116);
                    int16x4_t _S0high = vext_s16(_S0low, vget_high_s16(_S016), 3);
                    int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                    int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                    int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                    _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                    _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                    int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                    int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                    vst1_s16(rows0p, _rows0_sr4);
                    vst1_s16(rows1p, _rows1_sr4);
                } else if (channel == 4) {
                    int16x4_t _a0 = vdup_n_s16(a0);
                    int16x4_t _a1 = vdup_n_s16(a1);
                    uint8x8_t _S0 = vld1_u8(S0p);
                    uint8x8_t _S1 = vld1_u8(S1p);
                    int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                    int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                    int16x4_t _S0low = vget_low_s16(_S016);
                    int16x4_t _S1low = vget_low_s16(_S116);
                    int16x4_t _S0high = vget_high_s16(_S016);
                    int16x4_t _S1high = vget_high_s16(_S116);
                    int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                    int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                    _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                    _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                    int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                    int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                    vst1_s16(rows0p, _rows0_sr4);
                    vst1_s16(rows1p, _rows1_sr4);
                } else {
                    for (int dc = 0; dc < channel; ++dc) {
                        rows0p[dc] = (S0p[dc] * a0 + S0p[dc + channel] * a1) >> 4;
                        rows1p[dc] = (S1p[dc] * a0 + S1p[dc + channel] * a1) >> 4;
                    }
                }
#else
                for (int dc = 0; dc < channel; ++dc) {
                    rows0p[dc] = (S0p[dc] * a0 + S0p[dc + channel] * a1) >> 4;
                    rows1p[dc] = (S1p[dc] * a0 + S1p[dc + channel] * a1) >> 4;
                }
#endif

                ialphap += 2;
                rows0p += channel;
                rows1p += channel;
            }
        }

        prev_sy = sy;

        // vresize
        short b0 = ibeta[dy * 2];
        short b1 = ibeta[dy * 2 + 1];

        short *rows0p = rows0;
        short *rows1p = rows1;
        unsigned char *Dp = (unsigned char *)dst.data + dst_stride * dy;

#ifndef USE_NEON
        int remain = dsize.width * channel;
#else
        int nn = (dsize.width * channel) >> 3;
        int remain = (dsize.width * channel) - (nn << 3);
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn > 0; nn--) {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#endif
        for (; remain; --remain) {
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

int resize(const Mat &src, Mat &dst, const Size &dsize, int interpolation) {
    if (src.empty() || !dsize.area()) {
        LOGI("BSJ_AI::CV::resize src is empty or dsize.area == 0\n");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    double scale_x = (double)src.cols / dsize.width;
    double scale_y = (double)src.rows / dsize.height;
    Mat tmp = Mat(dsize.height, dsize.width, src.channels);

    if (interpolation == INTER_NEAREST) {
        ResizeNearestImpl(src, tmp, dsize, scale_x, scale_y);
    } else if (interpolation == INTER_LINEAR) {
        ResizeBilinearImpl(src, tmp, dsize, scale_x, scale_y);
    } else {
        LOGI("BSJ_AI::CV::resize not support interpolation %d\n", interpolation);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }
    dst.release();
    dst = tmp;
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int resizeYUV420sp(const Mat &src, Mat &dst, const Size &dsize, int interpolation) {
    if (src.empty() || !dsize.area() || src.channels != 1) {
        LOGI("BSJ_AI::CV::resize src is empty or dsize.area == 0 or src.channels != 1\n");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }
    double scale_x = 0.f;
    double scale_y = 0.f;
    Mat tmp = Mat(dsize.height + (dsize.height >> 1), dsize.width, src.channels);

    // resize Y
    unsigned char *srcY = (unsigned char *)src.data;
    unsigned char *dstY = (unsigned char *)tmp.data;
    int srcYHeight = int(src.rows * 2.f / 3.f);
    int dstYHeight = int(tmp.rows * 2.f / 3.f);

    Mat srcMatY(srcYHeight, src.cols, 1, srcY);
    Mat dstMatY(dstYHeight, tmp.cols, 1, dstY);

    scale_x = (double)srcMatY.cols / dstMatY.cols;
    scale_y = (double)srcMatY.rows / dstMatY.rows;
    if (interpolation == INTER_NEAREST) {
        ResizeNearestImpl(srcMatY, dstMatY, Size(dstMatY.cols, dstMatY.rows), scale_x, scale_y);
    } else if (interpolation == INTER_LINEAR) {
        ResizeBilinearImpl(srcMatY, dstMatY, Size(dstMatY.cols, dstMatY.rows), scale_x, scale_y);
    } else {
        LOGI("BSJ_AI::CV::resize not support interpolation %d\n", interpolation);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    // resize UV
    unsigned char *srcUV = srcY + srcMatY.rows * srcMatY.cols * srcMatY.channels;
    unsigned char *dstUV = dstY + dstMatY.rows * dstMatY.cols * dstMatY.channels;
    srcYHeight = int(src.rows / 3.f);
    dstYHeight = int(tmp.rows / 3.f);

    Mat srcMatUV(srcYHeight, src.cols / 2, 2, srcUV);
    Mat dstMatUV(dstYHeight, tmp.cols / 2, 2, dstUV);

    scale_x = (double)srcMatUV.cols / dstMatUV.cols;
    scale_y = (double)srcMatUV.rows / dstMatUV.rows;
    if (interpolation == INTER_NEAREST) {
        ResizeNearestImpl(srcMatUV, dstMatUV, Size(dstMatUV.cols, dstMatUV.rows), scale_x, scale_y);
    } else if (interpolation == INTER_LINEAR) {
        ResizeBilinearImpl(srcMatUV, dstMatUV, Size(dstMatUV.cols, dstMatUV.rows), scale_x, scale_y);
    } else {
        LOGI("BSJ_AI::CV::resize not support interpolation %d\n", interpolation);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }
    dst.release();
    dst = tmp;
    return BSJ_AI_FLAG_SUCCESSFUL;
}

}
} // namespace BSJ_AI::CV
