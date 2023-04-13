#ifndef _BSJ_AI_NEON_
#define _BSJ_AI_NEON_

#ifdef USE_NEON
    #ifdef __arm__
        #include <arm_neon.h>
    #else
        #include "ARM_NEON_2_x86_SSE/NEON_2_SSE.h"
    #endif // __arm__
#endif     // USE_NEON

namespace BSJ_AI {

    static void fill(unsigned char *data, int value, int length) {
        unsigned char *ptr = data;
#ifdef USE_NEON
        int       nn     = length >> 3;
        int       remain = length - (nn << 3);
        uint8x8_t _v     = vdup_n_u8(value);
        for (; nn > 0; nn--) {
            vst1_u8(ptr, _v);
            ptr += 8;
        }
#else
        int remain = length;
#endif // USE_NEON
        for (; remain > 0; remain--) {
            *ptr++ = value;
        }
    }

    static void fill(int *data, int value, int length) {
        int *ptr = data;
#ifdef USE_NEON
        int       nn     = length >> 2;
        int       remain = length - (nn << 2);
        int32x4_t _v     = vdupq_n_s32(value);
        for (; nn > 0; nn--) {
            vst1q_s32(ptr, _v);
            ptr += 8;
        }
#else
        int remain = length;
#endif // USE_NEON
        for (; remain > 0; remain--) {
            *ptr++ = value;
        }
    }

    static void fill(float *data, float value, int length) {
        float *ptr = data;
#ifdef USE_NEON
        int         nn     = length >> 2;
        int         remain = length - (nn << 2);
        float32x4_t _v     = vdupq_n_f32(value);
        for (; nn > 0; nn--) {
            vst1q_f32(ptr, _v);
            ptr += 8;
        }
#else
        int remain = length;
#endif // USE_NEON
        for (; remain > 0; remain--) {
            *ptr++ = value;
        }
    }
} // namespace BSJ_AI

#endif // !_BSJ_AI_NEON_
