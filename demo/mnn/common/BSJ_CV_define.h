#pragma once
#include <opencv2/core/version.hpp>

// #define CV_VERSION_EPOCH    2
// #define CV_VERSION_MAJOR    4
// #define CV_VERSION_MINOR    13
// #define CV_VERSION_REVISION 6

// #define CV_VERSION_MAJOR    4
// #define CV_VERSION_MINOR    5
// #define CV_VERSION_REVISION 2
//
// #define CV_VERSION_MAJOR    3
// #define CV_VERSION_MINOR    1
// #define CV_VERSION_REVISION 0

#define BSJ_CV_VERSION CV_MAJOR_VERSION

#if BSJ_CV_VERSION > 2
    #include <opencv2/calib3d/calib3d_c.h>
    #include <opencv2/imgproc/imgproc_c.h>
    #include <opencv2/core/types_c.h>
    #include <opencv2/core/core_c.h>
#else
    #include "opencv2/core.hpp"
    #include "opencv2/calib3d.hpp"
    #include "opencv2/imgproc.hpp"
#endif