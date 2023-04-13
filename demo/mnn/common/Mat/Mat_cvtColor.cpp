#include "Mat.h"

namespace BSJ_AI {
namespace CV {

// https://zh.wikipedia.org/wiki/YUV
/*
*   ISBN 1-878707-09-4
*          Y = 0.257R + 0.504G + 0.098B + 16
*   Cr  =  V = 0.439R ? 0.368G ? 0.071B + 128
*   Cb  =  U = ?0.148R ? 0.291G + 0.439B + 128
*/
//y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
//u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
//v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

/*
*  CCIR 601
*      Y = 0.299R + 0.587G + 0.114B
* Cr = V = 0.500R ? 0.419G ? 0.081B + 128
* Cb = U = ?0.169R ? 0.331G + 0.500B + 128
*/
//y = (77 * r + 150 * g + 29 * b) >> 8;
//u = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
//v = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
void C3_Y(int r, int g, int b, int& y) {
    y = (77 * r + 150 * g + 29 * b) >> 8;
}
void C3_U(int r, int g, int b, int& u) {
    u = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
}
void C3_V(int r, int g, int b, int& v) {
    v = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
}


#ifdef USE_NEON 
    uint8x8_t _u0 = vdup_n_u8(0);
    uint8x8_t _u77 = vdup_n_u8(77);
    uint8x8_t _u150 = vdup_n_u8(150);
    uint8x8_t _u29 = vdup_n_u8(29);
    int16x8_t _s43 = vdupq_n_s16(-43);
    int16x8_t _s85 = vdupq_n_s16(-85);
    int16x8_t _s107 = vdupq_n_s16(-107);
    int16x8_t _s21 = vdupq_n_s16(-21);
    int16x8_t _s128 = vdupq_n_s16(128);
    int8x8_t _i128 = vdup_n_s8(-128);
    int16x8_t _s0 = vdupq_n_s16(0);

    uint8x8_t _u128 = vdup_n_u8(128);
    int8x8_t _i90 = vdup_n_s8(90);
    int8x8_t _i46 = vdup_n_s8(46);
    int8x8_t _i22 = vdup_n_s8(22);
    int8x8_t _i113 = vdup_n_s8(113);

    // 因为uint8相加会越界，用int16代替会出现int16x16,超过128
    // 故改成8
void LOAD_BGR(unsigned char* src_pixel, uint8x16x3_t& _pixel, uint8x8_t& high_b, uint8x8_t& low_b, uint8x8_t& high_g, uint8x8_t& low_g, uint8x8_t& high_r, uint8x8_t& low_r)
{
    _pixel = vld3q_u8(src_pixel);

    high_b = vget_high_u8(_pixel.val[0]);
    low_b = vget_low_u8(_pixel.val[0]);
    high_g = vget_high_u8(_pixel.val[1]);
    low_g = vget_low_u8(_pixel.val[1]);
    high_r = vget_high_u8(_pixel.val[2]);
    low_r = vget_low_u8(_pixel.val[2]);
}

void LOAD_RGB(unsigned char* src_pixel, uint8x16x3_t& _pixel, uint8x8_t& high_b, uint8x8_t& low_b, uint8x8_t& high_g, uint8x8_t& low_g, uint8x8_t& high_r, uint8x8_t& low_r)
{
    _pixel = vld3q_u8(src_pixel);

    high_r = vget_high_u8(_pixel.val[0]);
    low_r = vget_low_u8(_pixel.val[0]);
    high_g = vget_high_u8(_pixel.val[1]);
    low_g = vget_low_u8(_pixel.val[1]);
    high_b = vget_high_u8(_pixel.val[2]);
    low_b = vget_low_u8(_pixel.val[2]);
}

void C3_Y_NEON(uint8x16_t& pixel_y, uint8x8_t high_b, uint8x8_t low_b, uint8x8_t high_g, uint8x8_t low_g, uint8x8_t high_r, uint8x8_t low_r)
{
    uint16x8_t high_y = vmull_u8(high_r, _u77);  // 乘法, 77 * R
    uint16x8_t low_y = vmull_u8(low_r, _u77);

    high_y = vmlal_u8(high_y, high_g, _u150);   // 乘加, 150 * G
    low_y = vmlal_u8(low_y, low_g, _u150);

    high_y = vmlal_u8(high_y, high_g, _u29);   // 乘加, 29 * B
    low_y = vmlal_u8(low_y, low_g, _u29);

    uint8x8_t u8_high_y = vshrn_n_u16(high_y, 8);    // 右移 8位
    uint8x8_t u8_low_y = vshrn_n_u16(low_y, 8);

    high_y = vaddl_u8(u8_high_y, _u0);    // 加法, +16
    low_y = vaddl_u8(u8_low_y, _u0);

    pixel_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y)); // 长度合并，两个长度8合并成16      
}

void C3_UV_NEON(uint16x8_t& unsigned_u, uint16x8_t& unsigned_v, uint8x8_t high_b, uint8x8_t low_b, uint8x8_t high_g, uint8x8_t low_g, uint8x8_t high_r, uint8x8_t low_r)
{
    uint8x8x2_t mix_r = vuzp_u8(low_r, high_r);
    uint8x8x2_t mix_g = vuzp_u8(low_g, high_g);
    uint8x8x2_t mix_b = vuzp_u8(low_b, high_b);

    int16x8_t signed_r = vreinterpretq_s16_u16(vaddl_u8(mix_r.val[0], _u0));
    int16x8_t signed_g = vreinterpretq_s16_u16(vaddl_u8(mix_g.val[0], _u0));
    int16x8_t signed_b = vreinterpretq_s16_u16(vaddl_u8(mix_b.val[0], _u0));

    int16x8_t signed_u = vmulq_s16(signed_r, _s43);                            // 乘法 -43 * R
    int16x8_t signed_v = vmulq_s16(signed_r, _s128);                           // 乘法 128 * R

    signed_u = vmlaq_s16(signed_u, signed_g, _s85);                            // 乘加, - 74 * G
    signed_v = vmlaq_s16(signed_v, signed_g, _s107);                           // 乘加, - 107 * G

    signed_u = vmlaq_s16(signed_u, signed_b, _s128);                           // 乘加, + 112 * G
    signed_v = vmlaq_s16(signed_v, signed_b, _s21);                            // 乘加, - 21 * G

    int8x8_t s8_u = vshrn_n_s16(signed_u, 8);                                  // 右移 8位
    int8x8_t s8_v = vshrn_n_s16(signed_v, 8);                                  // 右移 8位

    signed_u = vsubl_s8(s8_u, _i128);                                         // 减法， -（-128），也是加法
    signed_v = vsubl_s8(s8_v, _i128);                                         // 减法， -（-128），也是加法

    signed_u = vmaxq_s16(signed_u, _s0);                                  // 最大值, 和0比较
    signed_v = vmaxq_s16(signed_v, _s0);                                   // 最大值, 和0比较

    unsigned_u = vreinterpretq_u16_s16(signed_u);
    unsigned_v = vreinterpretq_u16_s16(signed_v);
}
#endif // USE_NEON

int fromBGRtoNV12(const Mat& src, Mat& dst) {
    int h = (src.rows >> 1 << 1);
    int w = (src.cols >> 1 << 1);

    dst.release();
    dst = Mat(h + (h >> 1), w, 1);
    unsigned char* yptr = (unsigned char*)dst.data;
    unsigned char* uvptr = (unsigned char*)(dst.data + w * h);

    /* Y Y
    *  Y Y
    *  U V
    */

    for (int row = 0; row < h; row++) {
        unsigned char* src_pixel = (unsigned char*)src.data + row * src.cols * src.channels;
#ifdef USE_NEON
        // 一次操作48个数据
        int nn = w >> 4;
        int remain = w - (nn << 4);
        for (; nn > 0; nn--) {
            uint8x16x3_t _pixel;
            uint8x16_t pixel_y;
            uint8x8_t high_b, low_b, high_g, low_g, high_r, low_r; 

            LOAD_BGR(src_pixel, _pixel, high_b, low_b, high_g, low_g, high_r, low_r);
            C3_Y_NEON(pixel_y, high_b, low_b, high_g, low_g, high_r, low_r);

            vst1q_u8(yptr, pixel_y);

            if (row % 2 == 0) {
                uint16x8_t unsigned_u, unsigned_v;
                C3_UV_NEON(unsigned_u, unsigned_v, high_b, low_b, high_g, low_g, high_r, low_r);

                uint8x8x2_t result;
                result.val[0] = vqmovn_u16(unsigned_u);
                result.val[1] = vqmovn_u16(unsigned_v);

                vst2_u8(uvptr, result);
                uvptr += 16;
            }
            src_pixel += 48;
            yptr += 16;
        }
#else
        int remain = w;
#endif // USE_NEON
        int b, g, r, y, u, v;

        for (; remain > 0; remain--) {
            b = src_pixel[0];
            g = src_pixel[1];
            r = src_pixel[2];
            C3_Y(b, g, r, y);

            yptr[0] = BSJ_BETWEEN(y, 0, 255);
            src_pixel += 3;
            yptr++;

            if (row % 2 != 0 || remain % 2 != 0) continue;
            C3_U(b, g, r, u);
            C3_V(b, g, r, v);
            uvptr[0] = u;
            uvptr[1] = v;
            uvptr += 2;
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromBGRtoNV21(const Mat& src, Mat& dst) {
    int h = (src.rows >> 1 << 1);
    int w = (src.cols >> 1 << 1);

    dst.release();
    dst = Mat(h + (h >> 1), w, 1);
    unsigned char* yptr = (unsigned char*)dst.data;
    unsigned char* vuptr = (unsigned char*)(dst.data + w * h);

    /* Y Y
    *  Y Y
    *  V U
    */

    for (int row = 0; row < h; row++) {
        unsigned char* src_pixel = (unsigned char*)src.data + row * src.cols * src.channels;
#ifdef USE_NEON
        // 一次操作48个数据
        int nn = w >> 4;
        int remain = w - (nn << 4);
        for (; nn > 0; nn--) {
            uint8x16x3_t _pixel;
            uint8x16_t pixel_y;
            uint8x8_t high_b, low_b, high_g, low_g, high_r, low_r;

            LOAD_BGR(src_pixel, _pixel, high_b, low_b, high_g, low_g, high_r, low_r);
            C3_Y_NEON(pixel_y, high_b, low_b, high_g, low_g, high_r, low_r);

            vst1q_u8(yptr, pixel_y);

            if (row % 2 == 0) {
                uint16x8_t unsigned_u, unsigned_v;
                C3_UV_NEON(unsigned_u, unsigned_v, high_b, low_b, high_g, low_g, high_r, low_r);

                uint8x8x2_t result;
                result.val[0] = vqmovn_u16(unsigned_v);
                result.val[1] = vqmovn_u16(unsigned_u);

                vst2_u8(vuptr, result);
                vuptr += 16;
            }
            src_pixel += 48;
            yptr += 16;
        }
#else
        int remain = w;
#endif // USE_NEON
        int b, g, r, y, u, v;

        for (; remain > 0; remain--) {
            b = src_pixel[0];
            g = src_pixel[1];
            r = src_pixel[2];
            C3_Y(b, g, r, y);

            yptr[0] = BSJ_BETWEEN(y, 0, 255);
            src_pixel += 3;
            yptr++;

            if (row % 2 != 0 || remain % 2 != 0) continue;
            C3_U(b, g, r, u);
            C3_V(b, g, r, v);
            vuptr[0] = v;
            vuptr[1] = u;
            vuptr += 2;
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromRGBtoNV12(const Mat& src, Mat& dst) {
    int h = (src.rows >> 1 << 1);
    int w = (src.cols >> 1 << 1);

    dst.release();
    dst = Mat(h + (h >> 1), w, 1);
    unsigned char* yptr = (unsigned char*)dst.data;
    unsigned char* uvptr = (unsigned char*)(dst.data + w * h);

    /* Y Y
    *  Y Y
    *  U V
    */

    for (int row = 0; row < h; row++) {
        unsigned char* src_pixel = (unsigned char*)src.data + row * src.cols * src.channels;
#ifdef USE_NEON
        // 一次操作48个数据
        int nn = w >> 4;
        int remain = w - (nn << 4);
        for (; nn > 0; nn--) {
            uint8x16x3_t _pixel;
            uint8x16_t pixel_y;
            uint8x8_t high_b, low_b, high_g, low_g, high_r, low_r;

            LOAD_RGB(src_pixel, _pixel, high_b, low_b, high_g, low_g, high_r, low_r);
            C3_Y_NEON(pixel_y, high_b, low_b, high_g, low_g, high_r, low_r);

            vst1q_u8(yptr, pixel_y);

            if (row % 2 == 0) {
                uint16x8_t unsigned_u, unsigned_v;
                C3_UV_NEON(unsigned_u, unsigned_v, high_b, low_b, high_g, low_g, high_r, low_r);

                uint8x8x2_t result;
                result.val[0] = vqmovn_u16(unsigned_u);
                result.val[1] = vqmovn_u16(unsigned_v);

                vst2_u8(uvptr, result);
                uvptr += 16;
            }
            src_pixel += 48;
            yptr += 16;
        }
#else
        int remain = w;
#endif // USE_NEON
        int b, g, r, y, u, v;

        for (; remain > 0; remain--) {
            r = src_pixel[0];
            g = src_pixel[1];
            b = src_pixel[2];
            C3_Y(b, g, r, y);

            yptr[0] = BSJ_BETWEEN(y, 0, 255);
            src_pixel += 3;
            yptr++;

            if (row % 2 != 0 || remain % 2 != 0) continue;
            C3_U(b, g, r, u);
            C3_V(b, g, r, v);
            uvptr[0] = u;
            uvptr[1] = v;
            uvptr += 2;
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromRGBtoNV21(const Mat& src, Mat& dst) {
    int h = (src.rows >> 1 << 1);
    int w = (src.cols >> 1 << 1);

    dst.release();
    dst = Mat(h + (h >> 1), w, 1);
    unsigned char* yptr = (unsigned char*)dst.data;
    unsigned char* vuptr = (unsigned char*)(dst.data + w * h);

    /* Y Y
    *  Y Y
    *  V U
    */

    for (int row = 0; row < h; row++) {
        unsigned char* src_pixel = (unsigned char*)src.data + row * src.cols * src.channels;
#ifdef USE_NEON
        // 一次操作48个数据
        int nn = w >> 4;
        int remain = w - (nn << 4);
        for (; nn > 0; nn--) {
            uint8x16x3_t _pixel;
            uint8x16_t pixel_y;
            uint8x8_t high_b, low_b, high_g, low_g, high_r, low_r;

            LOAD_RGB(src_pixel, _pixel, high_b, low_b, high_g, low_g, high_r, low_r);
            C3_Y_NEON(pixel_y, high_b, low_b, high_g, low_g, high_r, low_r);

            vst1q_u8(yptr, pixel_y);

            if (row % 2 == 0) {
                uint16x8_t unsigned_u, unsigned_v;
                C3_UV_NEON(unsigned_u, unsigned_v, high_b, low_b, high_g, low_g, high_r, low_r);

                uint8x8x2_t result;
                result.val[0] = vqmovn_u16(unsigned_v);
                result.val[1] = vqmovn_u16(unsigned_u);

                vst2_u8(vuptr, result);
                vuptr += 16;
            }
            src_pixel += 48;
            yptr += 16;
        }
#else
        int remain = w;
#endif // USE_NEON
        int b, g, r, y, u, v;

        for (; remain > 0; remain--) {
            r = src_pixel[0];
            g = src_pixel[1];
            b = src_pixel[2];
            C3_Y(b, g, r, y);

            yptr[0] = BSJ_BETWEEN(y, 0, 255);
            src_pixel += 3;
            yptr++;

            if (row % 2 != 0 || remain % 2 != 0) continue;
            C3_U(b, g, r, u);
            C3_V(b, g, r, v);
            vuptr[0] = v;
            vuptr[1] = u;
            vuptr += 2;
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromNV12toBGR(const Mat& src, Mat& dst) {
    int h = src.rows * 2.f / 3.f;
    int w = src.cols;

    dst.release();
    dst = Mat(h, w, 3);

    unsigned char* ptr = (unsigned char*)dst.data;
    unsigned char* yptr = (unsigned char*)src.data;
    unsigned char* uvptr = (unsigned char*)(src.data + w * h);

    for (int i = 0; i < h; i+= 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* dst_pixel0 = ptr;           // 同一行
        unsigned char* dst_pixel1 = ptr + w * 3; // 下一行是w*3
#ifdef USE_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);

        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(uvptr), _u128));
            int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
            int8x8_t _uu = _uuuuvvvv.val[0];
            int8x8_t _vv = _uuuuvvvv.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _i90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _i46);
            _g0 = vmlsl_s8(_g0, _uu, _i22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _i113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _i90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _i46);
            _g1 = vmlsl_s8(_g1, _uu, _i22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _i113);

            uint8x8x3_t _dst_pixel0;
            uint8x8x3_t _dst_pixel1;

            _dst_pixel0.val[0] = vqshrun_n_s16(_b0, 6);
            _dst_pixel0.val[1] = vqshrun_n_s16(_g0, 6);
            _dst_pixel0.val[2] = vqshrun_n_s16(_r0, 6);

            _dst_pixel1.val[0] = vqshrun_n_s16(_b1, 6);
            _dst_pixel1.val[1] = vqshrun_n_s16(_g1, 6);
            _dst_pixel1.val[2] = vqshrun_n_s16(_r1, 6);

            vst3_u8(dst_pixel0, _dst_pixel0);
            vst3_u8(dst_pixel1, _dst_pixel1);

            yptr0 += 8;
            yptr1 += 8;
            uvptr += 8;
            dst_pixel0 += 24;
            dst_pixel1 += 24;
        }
#else
        int remain = w;
#endif
        int u, v;
        for (; remain > 0; remain-=2) {
            u = uvptr[0] - 128;
            v = uvptr[1] - 128;

            int ruv = 90 * v;
            int guv = -45 * v - 22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            int y01 = yptr0[1] << 6;
            int y10 = yptr1[0] << 6;
            int y11 = yptr1[1] << 6;

            dst_pixel0[0] = BSJ_BETWEEN((y00 + buv) >> 6, 0, 255);
            dst_pixel0[1] = BSJ_BETWEEN((y00 + guv) >> 6, 0, 255);
            dst_pixel0[2] = BSJ_BETWEEN((y00 + ruv) >> 6, 0, 255);

            dst_pixel0[3] = BSJ_BETWEEN((y01 + buv) >> 6, 0, 255);
            dst_pixel0[4] = BSJ_BETWEEN((y01 + guv) >> 6, 0, 255);
            dst_pixel0[5] = BSJ_BETWEEN((y01 + ruv) >> 6, 0, 255);

            dst_pixel1[0] = BSJ_BETWEEN((y10 + buv) >> 6, 0, 255);
            dst_pixel1[1] = BSJ_BETWEEN((y10 + guv) >> 6, 0, 255);
            dst_pixel1[2] = BSJ_BETWEEN((y10 + ruv) >> 6, 0, 255);

            dst_pixel1[3] = BSJ_BETWEEN((y11 + buv) >> 6, 0, 255);
            dst_pixel1[4] = BSJ_BETWEEN((y11 + guv) >> 6, 0, 255);
            dst_pixel1[5] = BSJ_BETWEEN((y11 + ruv) >> 6, 0, 255);

            yptr0 += 2;
            yptr1 += 2;
            uvptr += 2;
            dst_pixel0 += 6;
            dst_pixel1 += 6;
        }
        yptr += (2 * w);
        ptr += (2 * 3 * w);
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromNV21toBGR(const Mat& src, Mat& dst) {
    int h = src.rows * 2.f / 3.f;
    int w = src.cols;

    dst.release();
    dst = Mat(h, w, 3);
    
    unsigned char* ptr = (unsigned char*)dst.data;
    unsigned char* yptr = (unsigned char*)src.data;
    unsigned char* vuptr = (unsigned char*)(src.data + w * h);

    for (int i = 0; i < h; i += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* dst_pixel0 = ptr;           // 同一行
        unsigned char* dst_pixel1 = ptr + w * 3; // 下一行是w*3
#ifdef USE_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);

        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(vuptr), _u128));
            int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
            int8x8_t _vv = _uuuuvvvv.val[0];
            int8x8_t _uu = _uuuuvvvv.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _i90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _i46);
            _g0 = vmlsl_s8(_g0, _uu, _i22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _i113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _i90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _i46);
            _g1 = vmlsl_s8(_g1, _uu, _i22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _i113);

            uint8x8x3_t _dst_pixel0;
            uint8x8x3_t _dst_pixel1;

            _dst_pixel0.val[0] = vqshrun_n_s16(_b0, 6);
            _dst_pixel0.val[1] = vqshrun_n_s16(_g0, 6);
            _dst_pixel0.val[2] = vqshrun_n_s16(_r0, 6);

            _dst_pixel1.val[0] = vqshrun_n_s16(_b1, 6);
            _dst_pixel1.val[1] = vqshrun_n_s16(_g1, 6);
            _dst_pixel1.val[2] = vqshrun_n_s16(_r1, 6);

            vst3_u8(dst_pixel0, _dst_pixel0);
            vst3_u8(dst_pixel1, _dst_pixel1);

            yptr0 += 8;
            yptr1 += 8;
            vuptr += 8;
            dst_pixel0 += 24;
            dst_pixel1 += 24;
        }
#else
        int remain = w;
#endif
        int u, v;
        for (; remain > 0; remain -= 2) {
            v = vuptr[0] - 128;
            u = vuptr[1] - 128;

            int ruv = 90 * v;
            int guv = -45 * v - 22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            int y01 = yptr0[1] << 6;
            int y10 = yptr1[0] << 6;
            int y11 = yptr1[1] << 6;

            dst_pixel0[0] = BSJ_BETWEEN((y00 + buv) >> 6, 0, 255);
            dst_pixel0[1] = BSJ_BETWEEN((y00 + guv) >> 6, 0, 255);
            dst_pixel0[2] = BSJ_BETWEEN((y00 + ruv) >> 6, 0, 255);

            dst_pixel0[3] = BSJ_BETWEEN((y01 + buv) >> 6, 0, 255);
            dst_pixel0[4] = BSJ_BETWEEN((y01 + guv) >> 6, 0, 255);
            dst_pixel0[5] = BSJ_BETWEEN((y01 + ruv) >> 6, 0, 255);

            dst_pixel1[0] = BSJ_BETWEEN((y10 + buv) >> 6, 0, 255);
            dst_pixel1[1] = BSJ_BETWEEN((y10 + guv) >> 6, 0, 255);
            dst_pixel1[2] = BSJ_BETWEEN((y10 + ruv) >> 6, 0, 255);

            dst_pixel1[3] = BSJ_BETWEEN((y11 + buv) >> 6, 0, 255);
            dst_pixel1[4] = BSJ_BETWEEN((y11 + guv) >> 6, 0, 255);
            dst_pixel1[5] = BSJ_BETWEEN((y11 + ruv) >> 6, 0, 255);

            yptr0 += 2;
            yptr1 += 2;
            vuptr += 2;
            dst_pixel0 += 6;
            dst_pixel1 += 6;
        }
        yptr += (2 * w);
        ptr += (2 * 3 * w);
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromNV12toRGB(const Mat & src, Mat & dst) {
    int h = src.rows * 2.f / 3.f;
    int w = src.cols;

    dst.release();
    dst = Mat(h, w, 3);

    unsigned char* ptr = (unsigned char*)dst.data;
    unsigned char* yptr = (unsigned char*)src.data;
    unsigned char* uvptr = (unsigned char*)(src.data + w * h);

    for (int i = 0; i < h; i += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* dst_pixel0 = ptr;           // 同一行
        unsigned char* dst_pixel1 = ptr + w * 3; // 下一行是w*3
#ifdef USE_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);

        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(uvptr), _u128));
            int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
            int8x8_t _uu = _uuuuvvvv.val[0];
            int8x8_t _vv = _uuuuvvvv.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _i90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _i46);
            _g0 = vmlsl_s8(_g0, _uu, _i22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _i113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _i90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _i46);
            _g1 = vmlsl_s8(_g1, _uu, _i22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _i113);

            uint8x8x3_t _dst_pixel0;
            uint8x8x3_t _dst_pixel1;

            _dst_pixel0.val[0] = vqshrun_n_s16(_r0, 6);
            _dst_pixel0.val[1] = vqshrun_n_s16(_g0, 6);
            _dst_pixel0.val[2] = vqshrun_n_s16(_b0, 6);

            _dst_pixel1.val[0] = vqshrun_n_s16(_r1, 6);
            _dst_pixel1.val[1] = vqshrun_n_s16(_g1, 6);
            _dst_pixel1.val[2] = vqshrun_n_s16(_b1, 6);

            vst3_u8(dst_pixel0, _dst_pixel0);
            vst3_u8(dst_pixel1, _dst_pixel1);

            yptr0 += 8;
            yptr1 += 8;
            uvptr += 8;
            dst_pixel0 += 24;
            dst_pixel1 += 24;
        }
#else
        int remain = w;
#endif
        int u, v;
        for (; remain > 0; remain -= 2) {
            u = uvptr[0] - 128;
            v = uvptr[1] - 128;

            int ruv = 90 * v;
            int guv = -45 * v - 22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            int y01 = yptr0[1] << 6;
            int y10 = yptr1[0] << 6;
            int y11 = yptr1[1] << 6;

            dst_pixel0[0] = BSJ_BETWEEN((y00 + ruv) >> 6, 0, 255);
            dst_pixel0[1] = BSJ_BETWEEN((y00 + guv) >> 6, 0, 255);
            dst_pixel0[2] = BSJ_BETWEEN((y00 + buv) >> 6, 0, 255);

            dst_pixel0[3] = BSJ_BETWEEN((y01 + ruv) >> 6, 0, 255);
            dst_pixel0[4] = BSJ_BETWEEN((y01 + guv) >> 6, 0, 255);
            dst_pixel0[5] = BSJ_BETWEEN((y01 + buv) >> 6, 0, 255);

            dst_pixel1[0] = BSJ_BETWEEN((y10 + ruv) >> 6, 0, 255);
            dst_pixel1[1] = BSJ_BETWEEN((y10 + guv) >> 6, 0, 255);
            dst_pixel1[2] = BSJ_BETWEEN((y10 + buv) >> 6, 0, 255);

            dst_pixel1[3] = BSJ_BETWEEN((y11 + ruv) >> 6, 0, 255);
            dst_pixel1[4] = BSJ_BETWEEN((y11 + guv) >> 6, 0, 255);
            dst_pixel1[5] = BSJ_BETWEEN((y11 + buv) >> 6, 0, 255);

            yptr0 += 2;
            yptr1 += 2;
            uvptr += 2;
            dst_pixel0 += 6;
            dst_pixel1 += 6;
        }
        yptr += (2 * w);
        ptr += (2 * 3 * w);
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromNV21toRGB(const Mat & src, Mat & dst) {
    int h = src.rows * 2.f / 3.f;
    int w = src.cols;

    dst.release();
    dst = Mat(h, w, 3);

    unsigned char* ptr = (unsigned char*)dst.data;
    unsigned char* yptr = (unsigned char*)src.data;
    unsigned char* vuptr = (unsigned char*)(src.data + w * h);

    for (int i = 0; i < h; i += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* dst_pixel0 = ptr;           // 同一行
        unsigned char* dst_pixel1 = ptr + w * 3; // 下一行是w*3
#ifdef USE_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);

        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(vuptr), _u128));
            int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
            int8x8_t _vv = _uuuuvvvv.val[0];
            int8x8_t _uu = _uuuuvvvv.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _i90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _i46);
            _g0 = vmlsl_s8(_g0, _uu, _i22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _i113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _i90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _i46);
            _g1 = vmlsl_s8(_g1, _uu, _i22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _i113);

            uint8x8x3_t _dst_pixel0;
            uint8x8x3_t _dst_pixel1;

            _dst_pixel0.val[0] = vqshrun_n_s16(_r0, 6);
            _dst_pixel0.val[1] = vqshrun_n_s16(_g0, 6);
            _dst_pixel0.val[2] = vqshrun_n_s16(_b0, 6);

            _dst_pixel1.val[0] = vqshrun_n_s16(_r1, 6);
            _dst_pixel1.val[1] = vqshrun_n_s16(_g1, 6);
            _dst_pixel1.val[2] = vqshrun_n_s16(_b1, 6);

            vst3_u8(dst_pixel0, _dst_pixel0);
            vst3_u8(dst_pixel1, _dst_pixel1);

            yptr0 += 8;
            yptr1 += 8;
            vuptr += 8;
            dst_pixel0 += 24;
            dst_pixel1 += 24;
        }
#else
        int remain = w;
#endif
        int u, v;
        for (; remain > 0; remain -= 2) {
            v = vuptr[0] - 128;
            u = vuptr[1] - 128;

            int ruv = 90 * v;
            int guv = -45 * v - 22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            int y01 = yptr0[1] << 6;
            int y10 = yptr1[0] << 6;
            int y11 = yptr1[1] << 6;

            dst_pixel0[0] = BSJ_BETWEEN((y00 + ruv) >> 6, 0, 255);
            dst_pixel0[1] = BSJ_BETWEEN((y00 + guv) >> 6, 0, 255);
            dst_pixel0[2] = BSJ_BETWEEN((y00 + buv) >> 6, 0, 255);

            dst_pixel0[3] = BSJ_BETWEEN((y01 + ruv) >> 6, 0, 255);
            dst_pixel0[4] = BSJ_BETWEEN((y01 + guv) >> 6, 0, 255);
            dst_pixel0[5] = BSJ_BETWEEN((y01 + buv) >> 6, 0, 255);

            dst_pixel1[0] = BSJ_BETWEEN((y10 + ruv) >> 6, 0, 255);
            dst_pixel1[1] = BSJ_BETWEEN((y10 + guv) >> 6, 0, 255);
            dst_pixel1[2] = BSJ_BETWEEN((y10 + buv) >> 6, 0, 255);

            dst_pixel1[3] = BSJ_BETWEEN((y11 + ruv) >> 6, 0, 255);
            dst_pixel1[4] = BSJ_BETWEEN((y11 + guv) >> 6, 0, 255);
            dst_pixel1[5] = BSJ_BETWEEN((y11 + buv) >> 6, 0, 255);

            yptr0 += 2;
            yptr1 += 2;
            vuptr += 2;
            dst_pixel0 += 6;
            dst_pixel1 += 6;
        }
        yptr += (2 * w);
        ptr += (2 * 3 * w);
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int fromBGRtoRGB(const Mat& src, Mat& dst) {
    int h = src.rows;
    int w = src.cols;

    dst.release();
    dst = Mat(h, w, 3);

    unsigned char* src_pixel = (unsigned char*)src.data;
    unsigned char* dst_pixel = (unsigned char*)dst.data;
    for (int row = 0; row < h; row++) {

#if USE_NEON
        // 一次操作16个数据
        int nn = w >> 4;
        int remain = w - (nn << 4);

        for (; nn > 0; nn--)
        {
            uint8x16x3_t _src_pixel = vld3q_u8(src_pixel);
            
            uint8x16x3_t _dst_pixel;
            _dst_pixel.val[0] = _src_pixel.val[2];
            _dst_pixel.val[1] = _src_pixel.val[1];
            _dst_pixel.val[2] = _src_pixel.val[0];

            vst3q_u8(dst_pixel, _dst_pixel);
            src_pixel += 3 * 16;
            dst_pixel += 3 * 16;
        }
#else
        int remain = w;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst_pixel[2] = BSJ_BETWEEN(src_pixel[0], 0, 255);
            dst_pixel[1] = BSJ_BETWEEN(src_pixel[1], 0, 255);
            dst_pixel[0] = BSJ_BETWEEN(src_pixel[2], 0, 255);

            src_pixel += 3;
            dst_pixel += 3;
        }
    }
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int cvtColor(const Mat& src, Mat& dst, ColorConversionType type) {

    int nResult = BSJ_AI_FLAG_SUCCESSFUL;
    switch (type) {
    case ColorConversionType::COLOR_CONVERT_NV12TOBGR:
        fromNV12toBGR(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_NV12TORGB:
        fromNV21toBGR(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_NV21TOBGR:
        fromNV12toRGB(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_NV21TORGB:
        fromNV21toRGB(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_BGRTONV12:
        fromBGRtoNV12(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_BGRTONV21:
        fromBGRtoNV21(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_RGBTONV12:
        fromRGBtoNV12(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_RGBTONV21:
        fromRGBtoNV21(src, dst);
        break;
    case ColorConversionType::COLOR_CONVERT_BGRTORGB:
    case ColorConversionType::COLOR_CONVERT_RGBTOBGR:
        fromBGRtoRGB(src, dst);
        break;
    default:
        nResult = BSJ_AI_FLAG_BAD_PARAMETER;
    }
    return nResult;
}

}
} // namespace BSJ_AI::CV