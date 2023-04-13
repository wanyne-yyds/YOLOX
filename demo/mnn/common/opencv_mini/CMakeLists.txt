
file(GLOB OPENCV_SRC ${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/*.h)
include_directories(${CMAKE_CURRENT_LIST_DIR})
# ----------------------------------------------------------------------------------
#         core
# ----------------------------------------------------------------------------------
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/core/include/opencv2/*.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/core/include/opencv2/core/*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/core/src/*)

include_directories(${CMAKE_CURRENT_LIST_DIR}/core/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/dynamicuda/include)

# ----------------------------------------------------------------------------------
#         imgproc
# ----------------------------------------------------------------------------------
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/imgproc/include/opencv2/*.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/imgproc/include/opencv2/imgproc)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/imgproc/src/avx/*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/imgproc/src/avx2/*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/imgproc/src/*.h)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/imgproc/src/*.cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/imgproc/include)

# ----------------------------------------------------------------------------------
#         highgui
# ----------------------------------------------------------------------------------
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/include/opencv2/*.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/include/opencv2/highgui/highgui*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/grfmt*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/bitstrm*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/precomp.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/utils.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/cap.cpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/cap_images.cpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/loadsave.cpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/utils.cpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/window.cpp)
if(WIN32)
    file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/cap_cmu.cpp)
    file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/cap_dshow.cpp)
    file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/highgui/src/window_w32.cpp)
    add_definitions(-DHAVE_DSHOW)
    add_definitions(-DHAVE_WIN32UI)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/highgui/include)
# ----------------------------------------------------------------------------------
#         flann
# ----------------------------------------------------------------------------------
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/flann/include/opencv2/*.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/flann/include/opencv2/flann/*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/flann/src/*)
include_directories(${CMAKE_CURRENT_LIST_DIR}/flann/include)

# ----------------------------------------------------------------------------------
#         features2d
# ----------------------------------------------------------------------------------
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/features2d/include/opencv2/*.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/features2d/include/opencv2/features2d/*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/features2d/src/*)
include_directories(${CMAKE_CURRENT_LIST_DIR}/features2d/include)

# ----------------------------------------------------------------------------------
#         calib3d
# ----------------------------------------------------------------------------------
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/calib3d/include/opencv2/*.hpp)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/calib3d/include/opencv2/calib3d/*)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/calib3d/src/*)
include_directories(${CMAKE_CURRENT_LIST_DIR}/calib3d/include)

# ----------------------------------------------------------------------------------
#         3rdparty
# ----------------------------------------------------------------------------------
add_definitions(-DHAVE_JPEG)
add_definitions(-DHAVE_PNG)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-but-set-parameter -Wmissing-declarations -Wunused -Wshadow -Wsign-compare -Wcast-align -Wshorten-64-to-32 -Wattributes")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-parameter -Wmissing-declarations -Wunused -Wshadow -Wsign-compare -Wcast-align -Wshorten-64-to-32 -Wattributes")

if(WIN32)
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-parameter -Wstrict-prototypes")# clang
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter -Wstrict-prototypes")# clang
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4013 /wd4018 /wd4101 /wd4244 /wd4267 /wd4715") # vs2005
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4013 /wd4018 /wd4101 /wd4244 /wd4267 /wd4715") # vs2005
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# ----------------------------------------------------------------------------------
#         zlib
# ----------------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/zlib)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/zlib/*.c)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/zlib/*.h)

# ----------------------------------------------------------------------------------
#         libjasper
# ----------------------------------------------------------------------------------
if(WIN32 AND NOT ANDROID)
  add_definitions(-DHAVE_JASPER)
  add_definitions(-DEXCLUDE_MIF_SUPPORT -DEXCLUDE_PNM_SUPPORT -DEXCLUDE_BMP_SUPPORT -DEXCLUDE_RAS_SUPPORT  -DEXCLUDE_JPG_SUPPORT -DEXCLUDE_PGX_SUPPORT)

  include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjasper)
  file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjasper/*.c)
  file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjasper/*.h)
  file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjasper/jasper/*.h)

  add_definitions(-DJAS_WIN_MSVC_BUILD)
endif()
# ----------------------------------------------------------------------------------
#         libjpeg
# ----------------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjpeg)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjpeg/*.c)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libjpeg/*.h)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(jcdctmgr.c PROPERTIES COMPILE_FLAGS "-O1")
endif()

# ----------------------------------------------------------------------------------
#         libpng
# ----------------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/libpng)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libpng/*.c)
file(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libpng/*.h)

if(ARM AND USE_NEON AND NOT WIN32)
  list(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libpng/arm/filter_neon.S)
  list(GLOB OPENCV_SRC ${OPENCV_SRC} ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libpng/ arm/arm_init.c)
  add_definitions(-DPNG_ARM_NEON_OPT=2)
else()
  add_definitions(-DPNG_ARM_NEON_OPT=0)
endif()

add_definitions(-DPNG_CONFIGURE_LIBPNG)

# append opencv mini source
list(APPEND BSJ_AI_SRC ${OPENCV_SRC})