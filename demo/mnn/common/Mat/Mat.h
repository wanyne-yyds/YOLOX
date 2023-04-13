#ifndef _MAT_
#define _MAT_
#include "BSJ_AI_defines.h"
#include "BSJ_AI_config.h"

#ifdef USE_NEON
#ifdef __arm__
#include <arm_neon.h>
#else
#include "ARM_NEON_2_x86_SSE/NEON_2_SSE.h"
#endif // __arm__
#endif // USE_NEON

#define BSJ_AI_CV_VERSION "v1.0.8.a.20230321"

namespace BSJ_AI {
namespace CV {
#define CV_PI 3.1415926535897932384626433832795

///////////////////////// memory align /////////////////////////////////////////
// from ncnn allocator.h
#if BSJ_AI_AVX512
#define BSJ_AI_MALLOC_ALIGN 64
#elif BSJ_AI_AVX
#define BSJ_AI_MALLOC_ALIGN 32
#else
#define BSJ_AI_MALLOC_ALIGN 16
#endif

#define BSJ_AI_MALLOC_OVERREAD 64

static inline int cv_xadd(int *addr, int delta) {
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
template <typename _Tp>
static inline _Tp *alignPtr(_Tp *ptr, int n = (int)sizeof(_Tp)) {
    return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

static inline size_t alignSize(size_t sz, int n) {
    return (sz + n - 1) & -n;
}

static inline void *fastMalloc(size_t size) {
#if _MSC_VER
    return _aligned_malloc(size, BSJ_AI_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void *ptr = 0;
    if (posix_memalign(&ptr, BSJ_AI_MALLOC_ALIGN, size + BSJ_AI_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
    unsigned char *udata = (unsigned char *)malloc(size + sizeof(void *) + BSJ_AI_MALLOC_ALIGN + BSJ_AI_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char **adata = alignPtr((unsigned char **)udata + 1, BSJ_AI_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static inline void fastFree(void *ptr) {
    if (ptr) {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char *udata = ((unsigned char **)ptr)[-1];
        free(udata);
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class Mat_ {
public:

    /*
     * 构造函数
     */
    Mat_(int _rows = 0, int _cols = 0, int _channels = 1, T *_data = 0);

    Mat_(const Size &size, int _channels = 1, T *_data = 0);

    Mat_(const Mat_<T> &A);

    /*
     * 析构函数
     */
    ~Mat_();

    /*
     * 释放指针
     */
    void release();

    /*
     * 判断图像是否为空
     */
    bool empty() const;

    /*
     * 设置全0的图像
     */
    bool setZeros();

    /*
     * 填充数据
     */
    bool fill(T value);

    bool fill(T value, int _start, int _end);

    bool fill(Scalar _value, Rect roi);

    size_t total() const;

    bool setEye();

    // bool diag();

    // determinant
    bool det(T &determinant) const;

    // 交换 r1 r2行
    bool SwapRow(int r1, int r2);

    /* Multiply row r of a matrix by a scalar.
    This is one of the three "elementary row operations". */
    // 第r行乘scalar
    bool ScaleRow(int r, T scalar);

    /* Add scalar * row r2 to row r1. */
    // r1行 =  r1 + r2行 * scalar
    bool ShearRow(int r1, int r2, T scalar) {
        if (r1 == r2) {
            return false; // 非法输入
        }

        for (int i = 0; i < this->cols; i++) {
            this->data[r1 * this->cols + i] += scalar * this->data[r2 * this->cols + i];
        }
        return true;
    }

    // clone
    Mat_<T> clone() const {
        Mat_<T> m(this->rows, this->cols, this->channels);
        if (m.data && this->data) {
            memcpy(m.data, this->data, this->total() * sizeof(T));
        }
        return m;
    }
    // inv
    Mat_<T> inv() const;

    // transpose
    Mat_<T> t() const;

    // a * aT
    Mat_<T> ATA() const;

    // operator
    Mat_<T> operator=(const Mat_<T> &A);

    // mul
    Mat_<T> operator*(const Mat_<T> &B);

    Mat_<T> operator*=(const T &b);       // mul
    Mat_<T> operator/=(const T &b);       // div
    Mat_<T> operator+=(const Mat_<T> &A); // add
    Mat_<T> operator-=(const Mat_<T> &A); // sub

    Mat_<T> operator()(const Rect &roi) const;

    // mean
    Mat_<T> mean(int axis = 0) const;
    // sum
    Mat_<T> sum(int axis = 0) const;

    int rows;
    int cols;
    int channels;
    int *refcount;
    T *data;

private:
    void init(int _rows, int _cols, int _channels, T *_data = 0);
};
typedef Mat_<int> Mat2i;
typedef Mat_<float> Mat2f;
typedef Mat_<double> Mat2d;
typedef Mat_<unsigned char> Mat;

enum InterpolationFlags {
    INTER_NEAREST = 0,
    INTER_LINEAR = 1
};

enum ColorConversionType {
    COLOR_CONVERT_NV12TOBGR = 0x00,
    COLOR_CONVERT_NV12TORGB = 0x01,
    COLOR_CONVERT_NV21TOBGR = 0x02,
    COLOR_CONVERT_NV21TORGB = 0x03,
    COLOR_CONVERT_BGRTONV12 = 0x04,
    COLOR_CONVERT_BGRTONV21 = 0x05,
    COLOR_CONVERT_RGBTONV12 = 0x06,
    COLOR_CONVERT_RGBTONV21 = 0x07,
    COLOR_CONVERT_BGRTORGB = 0x08,
    COLOR_CONVERT_RGBTOBGR = 0x10,
};
class ResizeNEAREST {
public:
    ResizeNEAREST(const IMAGE_FORMAT &format, int srcH, int srcW, int dstH, int dstW);
    ~ResizeNEAREST();

    int resize(const ImageData &srcData, ImageData &dstData);

private:
    void init(int srcH, int srcW, int dstH, int dstW);
    IMAGE_FORMAT m_eFormat;

    int m_nDstH;
    int m_nDstW;
    std::vector<int> m_arrX;
    std::vector<int> m_arrY;
};

// imgcodecs
Mat imread(const std::string &filename);

bool imwrite(const std::string &filename, const Mat &image);

Mat copyMakeBorder(Mat &image, int top, int bottom, int left, int right, int value = 0);

// cvtcolor
int cvtColor(const Mat &src, Mat &dst, ColorConversionType type);

// crop
int fromYuvRoi(const ImageData &inputData, Rect roi, ImageData &dst);

int fromYuvRoiResize(const ImageData &inputData, Rect roi, Size dstImageSize, ImageData &dst, int align);

bool cropImage(const Mat &img, const Rect &r, Mat &matCrop, const Size &resize_wh = Size(0, 0));

// draw
int circle(Mat &img, Point center, int radius, const Scalar &color, int thickness = -1);
int rectangle(Mat &img, Rect rect, const Scalar &color, int thickness = 1);
int rectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness = 1);
int draw_text(Mat &img, const std::string text, BSJ_AI::Point pt, const Scalar &color, int thickness = 5);
Size getTextSize(const std::string text, int thickness = 5);

// affine
Mat2f getRotationMatrix2D(Point center, double angle, double scale);
Mat2f getPerspectiveTransform(std::vector<Point2f> &src, std::vector<Point2f> &dst);
Mat2f getAffineTransform(std::vector<Point2f> &src, std::vector<Point2f> &dst);

int warpPerspective(const Mat &srcImage, Mat &dstImage, const Mat2f &M, const Size &dstSize);
// resize
int resize(const Mat &src, Mat &dst, const Size &dsize, int interpolation = INTER_LINEAR);
int resizeYUV420sp(const Mat &src, Mat &dst, const Size &dsize, int interpolation = INTER_LINEAR);

// filter
int Laplacian(const Mat &src, Mat &dst, int ksize = 1, double scale = 1, double delta = 0, int borderType = 0);

// fisheye
void initUndistortRectifyMap(Mat2f cameraMatrix, Mat2f distCoeffs, Mat2f R, Mat2f newCameraMatrix, Size size, Mat2f &map1, Mat2f &map2);
void undistortPoints(std::vector<Point2f> distorted, std::vector<Point2f> &undistorted, Mat2f cameraMatrix, Mat2f distCoeffs, Mat2f R, Mat2f newCameraMatrix);
}
} // namespace BSJ_AI::CV
#include "Mat.inl.h"
#endif // _MAT_
