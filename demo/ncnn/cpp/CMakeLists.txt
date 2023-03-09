cmake_minimum_required(VERSION 3.4.1)

option(USE_T7		"USE T7"            OFF)
option(USE_MNN      "USE MNN"           OFF)
option(USE_NCNN     "USE NCNN"          OFF)
option(USE_LINUX	"USE LINUX"         OFF)
option(USE_ROCKCHIP "USE ROCKCHIP"      ON)

message("==========================================================")
message("-- option:")
message("-- USE_T7                               is ${USE_T7}")
message("-- USE_MNN                              is ${USE_MNN}")
message("-- USE_NCNN                             is ${USE_NCNN}")
message("-- USE_LINUX                            is ${USE_LINUX}")
message("-- USE_ROCKCHIP                         is ${USE_ROCKCHIP}")
message("==========================================================")

project(HOD)

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
    if(USE_ROCKCHIP)
        set(TOOLCHAIN_DIR "/home/ckn/library/rknn_gcc/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin")
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/arm-linux-gnueabihf-g++)
        set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}/arm-linux-gnueabihf-gcc)
        set(CMAKE_C_FLAGS "-std=c++11 -O3 -fvisibility=hidden -fopenmp 
        -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections \
        -ffast-math -march=armv7-a -mfloat-abi=hard -mfpu=neon ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fvisibility=hidden -fopenmp \
        -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -march=armv7-a -mfloat-abi=hard -mfpu=neon -Wno-psabi ${CMAKE_CXX_FLAGS}")
        # OPENCV
        set(OPENCV_DIR "/home/ckn/library/opencv-3.4.16/install_rknn")
        # MNN
        set(MNN_DIR "~/library/MNN/build-arm-gnueabihf/rk_mnn")
        include_directories(${MNN_DIR}/include)
        link_directories(${MNN_DIR}/lib/)
        # NCNN
        set(NCNN_DIR "~/library/ncnn/build/arm-linux-gnueabihf/install")
        add_definitions(-DUSE_ROCKCHIP)

        # ROCHCHIP rknn api
		# set(RKNN_API_PATH ~/library/chip/rk/rknpu-master/rknn/rknn_api/librknn_api)
		# include_directories(${RKNN_API_PATH}/include)
		# link_directories(${RKNN_API_PATH}/lib/)
		# set(RKNN_API_LIB librknn_api.so)
		# ROCHCHIP rga
		# set(RGA_PATH  ~/Code/chip/rk/3rd/rga)
		# include_directories(${RGA_PATH})
		# include_directories(${RGA_PATH}/include)
		# link_directories(${RGA_PATH}/lib/)
		# set(RGA_LIBS librga.so)

        # set(libraries ${libraries} ${RKNN_API_LIB} ${RGA_LIBS})

    elseif(USE_T7)
        set(TOOLCHAIN_DIR "/opt/gcc-linaro-5.3.1-2016.05-x86_64_arm-linux-gnueabi/bin")
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/arm-linux-gnueabi-g++)
        set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}/arm-linux-gnueabi-gcc)
        set(CMAKE_C_FLAGS "-std=c++11 -O3 -fvisibility=hidden -fomit-frame-pointer \
        -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math \
        -march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fvisibility=hidden -fvisibility-inlines-hidden \
        -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math \
        -march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS} -lpthread -fopenmp")
        # OPENCV
        set(OPENCV_DIR "~/Code/tools/opencv-2.4.13.6/build-arm-linux-gnueabi/install")
        # MNN
        set(MNN_DIR "~/Code/tools/mnn")
        include_directories(${MNN_DIR}/include)
        link_directories(${MNN_DIR}/build/arm-linux-gnueabi/)
        # NCNN
        set(NCNN_DIR "~/Code/tools/ncnn/build-arm-linux-gnueabi/install")
    
    elseif(USE_LINUX)
        set(CMAKE_CXX_FLAGS "-std=c++11 -lpthread -fopenmp")
        # OPENCV
        set(OPENCV_DIR "~/Code/tools/opencv-2.4.13.6/build-linux/install")
        # MNN
        set(MNN_DIR "~/Code/tools/mnn")
        include_directories(${MNN_DIR}/include)
        link_directories(${MNN_DIR}/build/linux/)
        # NCNN
        set(NCNN_DIR "~/Code/tools/ncnn/build-linux/install")
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

add_executable(YOLOX ./yolox_h.cpp)
target_link_libraries(YOLOX ${NCNN_LIBS} ${MNN_LIBS} ${OpenCV_LIBS})