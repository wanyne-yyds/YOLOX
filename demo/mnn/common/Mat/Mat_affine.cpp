#include "Mat.h"
#include <cmath>

//#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
//#define SATURATE_CAST_INT(X) (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

namespace BSJ_AI {
namespace CV {
/*
 *       |   α   β     (1-α)tx - ty * β |
 * M =   |   -β   α    (1-α)ty + tx * β |
 * α = scale * cos angle
 * β = scale * sin angle
 */
Mat2f getRotationMatrix2D(Point center, double angle, double scale) {
    Mat2f M(2, 3, 1);

    angle *= (float)(3.14159265358979323846 / 180);
    float alpha = std::cos(angle) * scale;
    float beta = std::sin(angle) * scale;

    M.data[0] = alpha;
    M.data[1] = beta;
    M.data[2] = (1.f - alpha) * center.x - beta * center.y;
    M.data[3] = -beta;
    M.data[4] = alpha;
    M.data[5] = beta * center.x + (1.f - alpha) * center.y;

    return M;
}

/* Calculates coefficients of perspective transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   cij - matrix coefficients, c22 = 1
 */
Mat2f getPerspectiveTransform(std::vector<Point2f>& src, std::vector<Point2f>& dst)  {
    // CV_INSTRUMENT_REGION();

    // Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
    // double a[8][8], b[8];
    // Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    // for( int i = 0; i < 4; ++i )
    // {
    //     a[i][0] = a[i+4][3] = src[i].x;
    //     a[i][1] = a[i+4][4] = src[i].y;
    //     a[i][2] = a[i+4][5] = 1;
    //     a[i][3] = a[i][4] = a[i][5] =
    //     a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
    //     a[i][6] = -src[i].x*dst[i].x;
    //     a[i][7] = -src[i].y*dst[i].x;
    //     a[i+4][6] = -src[i].x*dst[i].y;
    //     a[i+4][7] = -src[i].y*dst[i].y;
    //     b[i] = dst[i].x;
    //     b[i+4] = dst[i].y;
    // }

    // static int param_IMGPROC_GETPERSPECTIVETRANSFORM_SOLVE_METHOD =
    //     (int)utils::getConfigurationParameterSizeT("OPENCV_IMGPROC_GETPERSPECTIVETRANSFORM_SOLVE_METHOD", (size_t)DECOMP_LU);
    // solve(A, B, X, param_IMGPROC_GETPERSPECTIVETRANSFORM_SOLVE_METHOD);
    // M.ptr<double>()[8] = 1.;

    // return M;
    return Mat2f();
}

/* 
 * https://en.wikipedia.org/wiki/Affine_transformation
 * Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 | |c01| |u1|
 * | x2 y2  1  0  0  0 | |c02| |u2|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * \  0  0  0 x2 y2  1 / |c12| |v2|
 *
 * where:
 *   cij - matrix coefficients
 */

Mat2f getAffineTransform(std::vector<Point2f>& src, std::vector<Point2f>& dst) {

    float a[6*6];
    float b[6];
    Mat2f A(6, 6, 1, a);
    Mat2f B(6, 1, 1, b);

    for( int i = 0; i < 3; i++ ) {
        int j = i * 12;
        int k = i * 12 + 6;

        a[j + 0] = a[k + 3] = src[i].x;
        a[j + 1] = a[k + 4] = src[i].y;
        a[j + 2] = a[k + 5] = 1;
        a[j + 3] = a[j + 4] = a[j + 5] = 0;
        a[k + 0] = a[k + 1] = a[k + 2] = 0;
        
        b[i * 2 + 0] = dst[i].x;
        b[i * 2 + 1] = dst[i].y;
    }

    /** 解线性方程 */
    Mat2f a_inv = A.inv();
    Mat2f X = a_inv * B;
    Mat2f M(2, 3);
    memcpy(M.data, X.data, sizeof(float) * X.total());
    return M;
}

/**
 *  有了仿射变换矩阵后，计算目的图像的公式为：
 *  dst(x,y)= src(M11x + M12y + M13, M21x + M22y + M23)
 * 
 */
int warpAffine(const Mat& src, const Mat& dst, const Mat2f& M, Size dsize, int flags) {
    if (src.empty() || M.empty() || (M.cols != 3 && M.rows != 2) || M.rows > 3) {
        LOGE("BSJ_AI::CV::warpAffine error src.empty() || M.empty() || (M.cols != 3 && M.rows !=2)  || M.rows > 3\n");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }
    
    Mat2f _M(3, 3, 1);
    _M.setEye();
    memcpy(_M.data, M.data, _M.total() * sizeof(float));

    std::vector<int> adelta(dsize.width);
    std::vector<int> bdelta(dsize.width);
    //for (int x = 0; x < dsize.width; x++) {
    //    adelta[x] = SATURATE_CAST_INT(_M.data[0] * x * (1 << 10));
    //    bdelta[x] = SATURATE_CAST_INT(_M.data[3] * x * (1 << 10));
    //}


    return BSJ_AI_FLAG_SUCCESSFUL;
}

int warpPerspective(const Mat &srcImage, Mat &dstImage, const Mat2f &M, const Size &dstSize) {
    if (srcImage.empty() || M.empty() || (M.cols != 3 && M.rows != 2) || M.rows > 3) {
        LOGE("BSJ_AI::CV::warpPerspective error srcImage.empty() || M.empty() || (M.cols != 3 && M.rows !=2)  || M.rows > 3\n");
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    Mat2f tmpM(3, 3, 1);
    tmpM.setEye();
    if (M.rows == 2) {
        memcpy(tmpM.data, M.data, M.rows * M.cols * M.channels * sizeof(float));
    } else {
        tmpM = M;
    }

    Mat2f invM = tmpM.inv();

    // warpPerspective
    dstImage = Mat(dstSize.height, dstSize.width, srcImage.channels);
    dstImage.setZeros();

    for (int row = 0; row < dstImage.rows; row++) {
        for (int col = 0; col < dstImage.cols; col++) {
            int x = invM.data[0] * col + invM.data[1] * row + invM.data[2];
            int y = invM.data[3] * col + invM.data[4] * row + invM.data[5];

            if (x >= 0 && x < srcImage.cols && y >= 0 && y < srcImage.rows) {
                int dIndex = (row * dstImage.cols + col) * dstImage.channels;
                int sIndex = (y * srcImage.cols + x) * srcImage.channels;
                for (int c = 0; c < dstImage.channels; c++) {
                    dstImage.data[dIndex + c] = srcImage.data[sIndex + c];
                }
            }
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

}
} // namespace BSJ_AI::CV