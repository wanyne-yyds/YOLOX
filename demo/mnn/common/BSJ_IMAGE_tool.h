#pragma once

#include <BSJ_AI_config.h>
#include <BSJ_CV_define.h>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

namespace BSJ_AI {
    typedef struct stRoiData {
        cv::Mat  image;
        cv::Rect roi;
    } ROI_DATA;

    static bool histogram(const cv::Mat             &image,
                          std::vector<unsigned int> &vecHistogram,
                          int                        nGroupExp         = 4,
                          int                        nSamplingInterval = 4,
                          int                        nChannel          = 0) {
        vecHistogram.clear();

        if (nGroupExp <= 0 || nSamplingInterval <= 0) {
            LOGE("BSJ_AI::histogram err: nGroupExp <= 0 || nSamplingInterval <= 0.\n");
            return false;
        }

        int nGroup = 256 >> nGroupExp;
        for (int i = 0; i < nGroup; i++) {
            vecHistogram.push_back(0);
        }

        int channels = image.channels();
        int rows     = image.rows;
        int cols     = image.cols * channels;

        if (rows * cols == 0) {
            LOGE("BSJ_AI::histogram err: rows * cols == 0.\n");
            return false;
        }

        if (image.isContinuous()) {
            cols *= rows;
            rows = 1;
        }

        int            i, j;
        unsigned char *p;
        for (i = 0; i < rows; i += nSamplingInterval) {
            p = (unsigned char *)image.ptr<unsigned char>(i);
            for (j = nChannel; j < cols; j += (nSamplingInterval * channels)) {
                int value = p[j] >> nGroupExp;
                vecHistogram[value]++;
            }
        }

        return true;
    }

    static bool enclosingRect(const std::vector<cv::Point> &points, cv::Rect &rect) {
        if (points.size() == 0) {
            LOGE("BSJ_AI::enclosingRect err: points.size() == 0.\n");
            return false;
        }

        int top    = points[0].y;
        int bottom = points[0].y;
        int left   = points[0].x;
        int right  = points[0].x;
        for (std::vector<cv::Point>::const_iterator it = points.begin() + 1; it != points.end(); it++) {
            if (it->x < left) {
                left = it->x;
            } else if (it->x > right) {
                right = it->x;
            }

            if (it->y < top) {
                top = it->y;
            } else if (it->y > bottom) {
                bottom = it->y;
            }
        }

        rect.x      = left;
        rect.width  = right - left + 1;
        rect.y      = top;
        rect.height = bottom - top + 1;

        return true;
    }

    // solve AX = b
    static bool solveEquations(const cv::Mat_<float> &matA, const cv::Mat_<float> &matB, cv::Mat &matX) {
        if (matA.rows != matB.rows) {
            LOGE("solveEquations err: matA.rows != matB.rows.\n");
            return false;
        } else if (matA.rows * matA.cols == 0) {
            LOGE("solveEquations err: matA.rows * matA.cols == 0.\n");
            return false;
        }

        //
        cv::Mat At = matA.t();
        cv::Mat T1 = At * matA;
        cv::Mat T2 = T1.inv();
        matX       = (T2 * At) * matB;

        return (cv::countNonZero(matX) > 0);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    //	bool polyfit(int, const std::vector<Point>&, std::vector<float>&)					//
    //	功能:																				//
    //		按关键字检索文件。																//
    //	返回值:																				//
    //		执行结果。																		//
    //			成功:	true																	//
    //			失败：	false																	//
    //	输入:																				//
    //		exponent			多项式次数													//
    //		points				点集														//
    //	输出:																				//
    //		polynomial			多项式系数[a0,a1,...,an,b], (a0 + a1x + ... + anx^n + by)	//
    //////////////////////////////////////////////////////////////////////////////////////////

    static bool polyfit(int exponent, const std::vector<cv::Point> &points, std::vector<float> &polynomial) {
        if (exponent < 1) {
            LOGE("BSJ_AI::polyfit err: exponent = %d.\n", exponent);
            return false;
        } else if (points.size() < 2) {
            LOGE("BSJ_AI::polyfit err: points.size() = %zd.\n", points.size());
            return false;
        }

        // Y = [y1; y2; ...; yn]
        cv::Mat Y = cv::Mat(points.size(), 1, CV_32FC1);
        // X = [1 x1 x1^2 ... x1^m; 1 x2 x2^2 ... x2^m; ... ...; 1 xn xn^2 ... xn^m]
        cv::Mat X = cv::Mat(points.size(), exponent + 1, CV_32FC1);

        for (int i = points.size() - 1; i >= 0; i--) {
            Y.at<float>(i, 0) = points[i].y;
            X.at<float>(i, 0) = 1;
            for (int j = 1; j <= exponent; j++) {
                X.at<float>(i, j) = X.at<float>(i, j - 1) * points[i].x;
            }
        }

        polynomial.clear();
        // XA = Y
        cv::Mat A;
        bool    bResult = solveEquations(X, Y, A);

        if (bResult) {
            for (int i = 0; i < A.rows; i++) {
                polynomial.push_back(A.at<float>(i, 0));
            }
            polynomial.push_back(-1);
            return true;
        } else {
            if (exponent == 1) {
                polynomial.push_back(points[0].x);
                polynomial.push_back(-1);
                polynomial.push_back(0);
                return true;
            } else {
                LOGE("BSJ_AI::polyfit err: Data induced degenerate matrix.\n");
                return false;
            }
        }
    }

    static bool fillConvexPoly(cv::Mat &matImage, const std::vector<cv::Point> &vecApex, const cv::Scalar &scalar, int begin = 0, int end = -1) {
        if (end < 0) {
            end = vecApex.size() - 1;
        }

        begin = BSJ_MAX(0, begin);
        end   = BSJ_MIN(vecApex.size(), end);

        int        sz      = end - begin + 1;
        cv::Point *aptApex = (cv::Point *)malloc(sz * sizeof(cv::Point));
        if (!aptApex) {
            LOGE("BSJ_AI::fillConvexPoly err: out of memory!\n");
            return false;
        }

        for (int i = 0; i < sz; i++) {
            aptApex[i] = vecApex[begin + i];
        }

        cv::fillConvexPoly(matImage, aptApex, sz, scalar);

        free(aptApex);

        return true;
    }

    static bool findChessboardCorners(const cv::Mat &matBGR, const cv::Size &szBoard, std::vector<cv::Point2f> &vecCorners) {
        bool bResult = cv::findChessboardCorners(matBGR, szBoard, vecCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
        if (bResult) {
            cv::Mat matGray;
            cv::cvtColor(matBGR, matGray, CV_BGR2GRAY);

            /* 亚像素精确化 */
            // find4QuadCornerSubpix(view_gray, image_points, Size(5, 5)); //对粗提取的角点进行精确化
            cv::cornerSubPix(matGray, vecCorners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.01));
            matGray.release();
        }

        return bResult;
    }

    //////////////////////////////////////////////////////////////
    //	bool deleteTail(cv::Mat&, int)							//
    //	功能:													//
    //		数据去尾。											//
    //	返回值:													//
    //		执行结果。											//
    //			成功:	true									//
    //			失败：	false									//
    //	输入:													//
    //		matImage			原始图像						//
    //		tailLen				去尾位长度，取值[0,8]			//
    //	输出:													//
    //		matImage			去尾图像						//
    //////////////////////////////////////////////////////////////

    static bool deleteTail(cv::Mat &matImage, int tailLen) {
        if (matImage.depth() != CV_8U) {
            return false;
        }

        // Mask 8bit
        unsigned char byteMask = 0x00;
        switch (tailLen) {
        case 0:
            return true;
        case 1: byteMask = 0xFE; break;
        case 2: byteMask = 0xFC; break;
        case 3: byteMask = 0xF8; break;
        case 4: byteMask = 0xF0; break;
        case 5: byteMask = 0xE0; break;
        case 6: byteMask = 0xC0; break;
        case 7: byteMask = 0x80; break;
        case 8:
            memset(matImage.data, 0x00, matImage.rows * matImage.cols * matImage.channels());
            return true;
        default:
            return false;
        }

        int dataLen = matImage.rows * matImage.cols * matImage.channels();
        // Mask 64bit
        uint64_t int64Mask = 0;
        for (int i = 0; i < 8; i++) {
            int64Mask |= (((uint64_t)byteMask) << (i * 8));
        }

        for (int i = 0; i < dataLen; i += 8) {
            uint64_t *value = (uint64_t *)(matImage.data + i);
            *value &= int64Mask;
        }
        for (int i = dataLen - dataLen % 8; i < dataLen; i++) {
            matImage.data[i] &= byteMask;
        }

        return true;
    }

    static bool cropImage(const cv::Mat &img, const cv::Rect &r, cv::Mat &matCrop, const cv::Size &resize_wh = cv::Size(0, 0)) {
        if (r.width <= 0 || r.height <= 0) {
            return false;
        }

        cv::Rect rect_image(0, 0, img.cols, img.rows);

        if (r == (r & rect_image) && (resize_wh.width > 0 || resize_wh.height > 0)) {
            cv::resize(img(r), matCrop, resize_wh);
        } else if (resize_wh.width > 0 || resize_wh.height > 0) {
            int                      wofs = r.width / resize_wh.width / 2;
            int                      hofs = r.height / resize_wh.height / 2;
            std::vector<cv::Point2f> src_pts(3);
            src_pts[0] = cv::Point2f(r.x + wofs, r.y + hofs);
            src_pts[1] = cv::Point2f(r.x + r.width - 1 - wofs, r.y + hofs);
            src_pts[2] = cv::Point2f(r.x + wofs, r.y + r.height - 1 - hofs);

            std::vector<cv::Point2f> dst_pts(3);
            dst_pts[0]        = cv::Point2f(0, 0);
            dst_pts[1]        = cv::Point2f(resize_wh.width - 1, 0);
            dst_pts[2]        = cv::Point2f(0, resize_wh.height - 1);
            cv::Mat rotateMat = cv::getAffineTransform(src_pts, dst_pts);

            int flag = cv::INTER_LINEAR;
            cv::warpAffine(img, matCrop, rotateMat, resize_wh, flag);
        } else {
            cv::Rect rect = r;

            if (matCrop.type() != img.type() || matCrop.rows != rect.height || matCrop.cols != rect.width) {
                matCrop = cv::Mat(rect.height, rect.width, img.type());
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
                img(rect).copyTo(matCrop(cv::Range(dy, dy + rect.height), cv::Range(dx, dx + rect.width)));
            }
        }

        if (!matCrop.isContinuous()) {
            matCrop = matCrop.clone();
        }

        return true;
    }

    //////////////////////////////////////////////////////////////
    //	float closeToBoundary(const cv::Rect&, int, int)		//
    //	功能:													//
    //		计算矩形框与图像边缘的距离偏移量。					//
    //		如果框贴边返回0，超出为负，图像内为正。				//
    //	返回值:													//
    //		最小偏移量。										//
    //	输入:													//
    //		rect			矩形框								//
    //		imgWidth		图像宽								//
    //		imgHeight		图像高								//
    //	输出:													//
    //		无													//
    //////////////////////////////////////////////////////////////

    static float closeToBoundary(const cv::Rect &rect, int imgWidth, int imgHeight) {
        if (imgWidth <= 0 || imgHeight <= 0) {
            return 0.f;
        }

        int   dTop    = rect.y;
        int   dBottom = imgHeight - (rect.y + rect.height);
        float xOffset = BSJ_MIN(dTop, dBottom) / (float)imgHeight;

        int   dLeft   = rect.x;
        int   dRight  = imgWidth - (rect.x + rect.width);
        float yOffset = BSJ_MIN(dLeft, dRight) / (float)imgWidth;

        return BSJ_MIN(xOffset, yOffset);
    }

} // namespace BSJ_AI