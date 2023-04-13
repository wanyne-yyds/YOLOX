#ifndef _INTERENCE_H_
#define _INTERENCE_H_

#include <iostream>
#include <memory>
#include <vector>
#include <mutex>
#include <functional>
#include "BSJ_AI_config.h"
#include "BSJ_AI_defines.h"
#include "BSJ_AI_function.h"
#include "Mat/Mat.h"

#define BSJ_AI_INTERENCE_VERSION "v3.0.1.a.20230316"

#ifdef USE_NCNN
    #include "ncnn/net.h"
    #include "ncnn/layer.h"
#endif // USE_NCNN

#ifdef USE_MNN
    #include "MNN/Interpreter.hpp"
    #include "MNN/MNNDefine.h"
    #include "MNN/Tensor.hpp"
    #include "MNN/ImageProcess.hpp"
#endif // USE_MNN

#ifdef USE_ROCKCHIP
    #include "rknn_api.h"
#endif

#ifdef USE_SIGMASTAR
    #include "mi_common_datatype.h"
    #include "mi_sys_datatype.h"
    #include "mi_ipu.h"
    #include "mi_sys.h"
#endif

#define BSJ_AI_MODEL_FORWARD_ASSERT(x, s)                                \
    {                                                                    \
        int         res = (x);                                           \
        std::string s1  = (s);                                           \
        if (res) {                                                       \
            LOGE("BSJ_AI::Inference file %s err func %s, line %d: %s\n", \
                 __FILE__, __func__, __LINE__, s1.c_str());              \
            return -1;                                                   \
        }                                                                \
    }

namespace BSJ_AI {
    class Inference {
    public:
        enum InferenceType {
            FORWARD_NCNN      = 0, /*<! use ncnn */
            FORWARD_MNN       = 1, /*<! use mnn */
            FORWARD_ROCKCHIP  = 2, /*<! use rockchip, 设备必须是瑞芯微的产品, eg.rv1126 rv1108 */
            FORWARD_SIGMASTAR = 3  /*<! use sigmastar, 设备必须是星辰的产品, eg.ss30kq or other */
        };

        struct Config {
            std::string model_path; /*<! 加载模型文件，如：.bin, .rknn, .mnn .img */
            std::string param_path; /*<! 加载模型配置文件，目前只有ncnn用到 */

            unsigned char *model = NULL; /*<! 加载模型头文件，一般是二进制，传进来的是一个static const 的数据，一般是放在rodata上 */
            unsigned char *param = NULL; /*<! 加载模型头文件，一般是二进制，传进来的是一个static const 的数据，一般是放在rodata上，目前只有ncnn用到 */
            int            nModelSize;   /*<! 对应model指针的长度 */

            std::vector<int>                      inpNodes;  /*<! 模型输入的节点，int */
            std::vector<std::string>              sInpNodes; /*<! 模型输入的节点, string */
            std::vector<std::vector<int>>         oupNodes;  /*<! 模型输出的节点，int */
            std::vector<std::vector<std::string>> sOupNodes; /*<! 模型输出的节点, string */

            InferenceType forward_type = InferenceType::FORWARD_NCNN; /*<! 定义推理方式, 默认是ncnn */

            IMAGE_FORMAT srcFormat  = IMAGE_FORMAT::BGR888;                         /*<! 定义模型输入数据类型， 默认是bgr888*/
            int          filterType = BSJ_AI::CV::InterpolationFlags::INTER_LINEAR; /*<! 图像压缩方式, 默认是双线性压缩 */

            int nThread = 1; /*<! 运行线程数量，使用npu时，用不到 */

            int netWidth  = 0; /*<! 输入尺寸，宽 */
            int netHeight = 0; /*<! 输入尺寸，高 */

            float mean[3]   = {127.5f, 127.5f, 127.5f};                /*<! 模型预处理 数据预处理方式， 减 */
            float normal[3] = {0.00784314f, 0.00784314f, 0.00784314f}; /*<! 模型预处理 数据预处理方式， 乘 */

            float              thresh   = 0.5f;                    /*<! 模型后处理 阈值*/
            int                nClasses = 1;                       /*<! 模型后处理 类别*/
            std::vector<float> strides  = std::vector<float>{1.f}; /*<! 模型后处理 下采样大小*/

            // 零拷贝数据，一般从vi传入数据
            bool use_zero_copy = false;
        };

        struct NCHW {
            int n;
            int c;
            int h;
            int w;

            NCHW(int _n = 0, int _c = 0, int _h = 0, int _w = 0) {
                n = _n;
                c = _c;
                h = _h;
                w = _w;
            }
            int size() {
                return n * c * h * w;
            }
        };

    public:
        // index data指针 shape维度
        typedef std::function<void(int, std::vector<float *>, std::vector<NCHW> &)> CallBack;

        /**
         * @brief 构造函数
         */
        Inference();

        /**
         * @brief 析构函数
         */
        ~Inference();

        /**
         * @brief 初始化参数
         * @author tsd
         * @param[in] cfg 配置参数
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int init(const Config &cfg);

        /**
         * @brief 执行推理
         * @author tsd
         * @param[in]   inputData       图像数据
         * @param[out]  outputCallBack  回调函数，需要自己处理
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int run(const ImageData &inputData, CallBack &outputCallBack);

        /**
         * @brief 执行推理，可用于关键点回归
         * @author tsd
         * @param[in]   inputData   图像数据
         * @param[out]  vecOutputs  输出数据，两个维度的数据
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int run(const ImageData &inputData, std::vector<std::vector<float>> &vecOutputs);

        /**
         * @brief 执行推理，适用于一个输出，比如分类，人脸识别
         * @author tsd
         * @param[in]   inputData   图像数据
         * @param[out]  vecOutputs  输出数据，两个维度的数据
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int run(const ImageData &inputData, std::vector<float> &vecOutputs);

        /**
         * @brief 执行推理，适用于一个输出，比如二分类，三分类
         * @author tsd
         * @param[in]   inputData   图像数据
         * @param[out]  score       分数
         * @param[out]  label       类别
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int run(const ImageData &inputData, float &score, int &label);

    private:
        /**
         * @brief 压缩数据，可以压缩bgr,rgb, nv12, nv21
         * @author tsd
         * @param[in]   src     图像数据
         * @param[out]  image   已经压缩的数据
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int resizeImage(const ImageData &src, CV::Mat &image);

        /**
         * @brief 转换数据,将输入的图像转换成模型需要的格式，需要在初始化的时候，确定srcFormat
         * @author tsd
         * @param[in]   src     图像数据
         * @param[in]   format  需要转换成的格式
         * @param[out]  dst     转换好的图像数据
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        void convertImage(const CV::Mat &src, CV::Mat &dst, IMAGE_FORMAT format);

    private:
        /**
         * @brief 初始化指针
         * @author tsd
         */
        void create_mnn();
        void create_ncnn();
        void create_rockchip();
        void create_sigmastar();

        /**
         * @brief 释放
         * @author tsd
         */
        void destory();
        void destory_mnn();
        void destory_ncnn();
        void destory_rockchip();
        void destory_sigmastar();

        /**
         * @brief 初始化模型
         * @author tsd
         * @param[in] cfg 配置参数
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int init_mnn(const Config &cfg);
        int init_ncnn(const Config &cfg);
        int init_rockchip(const Config &cfg);
        int init_sigmastar(const Config &cfg);

        /**
         * @brief 执行推理
         * @author tsd
         * @param[in]   image       图像数据
         * @param[out]  outputCallBack  回调函数，需要自己处理
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) 成功
         *      @retval BSJ_AI_FLAG_FAILED      (-2) 失败
         */
        int run_mnn(const CV::Mat &image, CallBack &outputCallBack);
        int run_ncnn(const CV::Mat &image, CallBack &outputCallBack);
        int run_rockchip(const CV::Mat &image, CallBack &outputCallBack);
        int run_sigmastar(const CV::Mat &image, CallBack &outputCallBack);

    private:
        std::mutex lock; /*<! 执行锁 */

#ifdef USE_NCNN
        std::shared_ptr<ncnn::Net> m_hNcnn;
#endif // USE_NCNN

#ifdef USE_MNN
        std::shared_ptr<MNN::Interpreter> m_hMnn;
        MNN::Session                     *session;

#endif // USE_MNN

#ifdef USE_ROCKCHIP
        rknn_context          m_hContext;
        rknn_input_output_num m_nIoNumber;

        rknn_tensor_attr *m_aInputAttr;
        rknn_tensor_attr *m_aOutputAttr;
        rknn_input       *m_aInputData;
        rknn_output      *m_aOutputData;
#endif // USE_ROCKCHIP

#ifdef USE_SIGMASTAR
        MI_U32                          m_hU32ChannelID = 65535;
        MI_IPU_SubNet_InputOutputDesc_t m_nDesc;
        MI_IPU_TensorVector_t           m_InputTensorVector;
        MI_IPU_TensorVector_t           m_OutputTensorVector;
#endif // USE_SIGMASTAR

        Config m_stCfg;
    };
} // namespace BSJ_AI

#endif // !_INTERENCE_H_