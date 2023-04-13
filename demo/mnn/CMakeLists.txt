cmake_minimum_required(VERSION 3.4.1)

option(USE_MNN      "USE MNN"           ON)
option(USE_NCNN     "USE NCNN"          OFF)
option(USE_LINUX	"USE LINUX"         ON)

message("==========================================================")
message("-- option:")
message("-- USE_MNN                              is ${USE_MNN}")
message("-- USE_NCNN                             is ${USE_NCNN}")
message("-- USE_LINUX                            is ${USE_LINUX}")
message("==========================================================")

project(BSD)

# set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(libraries )

# windows
if (CMAKE_HOST_WIN32)
	#opencv
	set(OPENCV_DIR "E:/MyDrivers/software/opencv/build")
	include_directories(${OPENCV_DIR}/include/)
	link_directories(${OPENCV_DIR}/x64/vc15/lib/)
	set(OpenCV_LIBS opencv_world3410.lib)
	# set(OpenCV_LIBS opencv_world3410d.lib)
	#ncnn
	set(NCNN_DIR "E:/MyDrivers/software/cnn_deploy/ncnn/install")
	include_directories(${NCNN_DIR}/include/)
	link_directories(${NCNN_DIR}/lib/)
	set(NCNN_LIBS ncnn.lib)
	# set(NCNN_LIBS ncnnd.lib)
# linux
elseif (CMAKE_HOST_UNIX)
    if(USE_LINUX)
        set(CMAKE_CXX_FLAGS "-std=c++11 -lpthread -fopenmp")
        # OPENCV
        set(OPENCV_DIR "~/library/opencv-3.4.16/build_linux/install")
        # MNN
        set(MNN_DIR "~/library/MNN")
        include_directories(${MNN_DIR}/include)
        link_directories(${MNN_DIR}/build_static/)
        # NCNN
        set(NCNN_DIR "~/library/ncnn/build/linux/install")
    endif()
    
    include_directories(${OPENCV_DIR}/include)
    link_directories(${OPENCV_DIR}/lib)
    set(OpenCV_LIBS libopencv_calib3d.so
        #libopencv_contrib.so
        libopencv_core.so
        libopencv_features2d.so
        libopencv_flann.so
        #libopencv_gpu.so
        libopencv_highgui.so
        libopencv_imgproc.so
        #libopencv_legacy.so
        libopencv_ml.so
        libopencv_objdetect.so
        #libopencv_ocl.so
        libopencv_photo.so
        # libopencv_stitching.so
        # libopencv_superres.so
        libopencv_video.so
        libopencv_imgcodecs.so
        # libopencv_videostab.so
        )

    include_directories(${NCNN_DIR}/include)
    link_directories(${NCNN_DIR}/lib/)
    set(NCNN_LIBS libncnn.a)
    set(MNN_LIBS libMNN.a)
endif()

file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/ImageManager/*)
# if (CMAKE_HOST_UNIX)
#     file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/ImageManagerRga/*)
# endif()
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/ARM_NEON_2_x86_SSE/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/BackgroundModeling/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/CameraCoords/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/Detector/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/MD5/*)
# file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/Store/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/Mat/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/Mat/exif/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/Mat/std/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/Inference/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/StatisticalQueue/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/ThreadSafeQueue/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/common/json/*)
file(GLOB BSD_SRC ${BSD_SRC} ${CMAKE_CURRENT_LIST_DIR}/example/*)

if(USE_MNN)
    add_definitions(-DUSE_MNN)
elseif(USE_NCNN)
    add_definitions(-DUSE_NCNN)
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${BSD_SRC})

include_directories("./common")
add_executable(BSD ${BSD_SRC})

target_link_libraries(BSD ${libraries} ${MNN_LIBS} ${NCNN_LIBS} ${OpenCV_LIBS})