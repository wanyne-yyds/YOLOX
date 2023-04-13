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
            FORWARD_ROCKCHIP  = 2, /*<! use rockchip, �豸��������о΢�Ĳ�Ʒ, eg.rv1126 rv1108 */
            FORWARD_SIGMASTAR = 3  /*<! use sigmastar, �豸�������ǳ��Ĳ�Ʒ, eg.ss30kq or other */
        };

        struct Config {
            std::string model_path; /*<! ����ģ���ļ����磺.bin, .rknn, .mnn .img */
            std::string param_path; /*<! ����ģ�������ļ���Ŀǰֻ��ncnn�õ� */

            unsigned char *model = NULL; /*<! ����ģ��ͷ�ļ���һ���Ƕ����ƣ�����������һ��static const �����ݣ�һ���Ƿ���rodata�� */
            unsigned char *param = NULL; /*<! ����ģ��ͷ�ļ���һ���Ƕ����ƣ�����������һ��static const �����ݣ�һ���Ƿ���rodata�ϣ�Ŀǰֻ��ncnn�õ� */
            int            nModelSize;   /*<! ��Ӧmodelָ��ĳ��� */

            std::vector<int>                      inpNodes;  /*<! ģ������Ľڵ㣬int */
            std::vector<std::string>              sInpNodes; /*<! ģ������Ľڵ�, string */
            std::vector<std::vector<int>>         oupNodes;  /*<! ģ������Ľڵ㣬int */
            std::vector<std::vector<std::string>> sOupNodes; /*<! ģ������Ľڵ�, string */

            InferenceType forward_type = InferenceType::FORWARD_NCNN; /*<! ��������ʽ, Ĭ����ncnn */

            IMAGE_FORMAT srcFormat  = IMAGE_FORMAT::BGR888;                         /*<! ����ģ�������������ͣ� Ĭ����bgr888*/
            int          filterType = BSJ_AI::CV::InterpolationFlags::INTER_LINEAR; /*<! ͼ��ѹ����ʽ, Ĭ����˫����ѹ�� */

            int nThread = 1; /*<! �����߳�������ʹ��npuʱ���ò��� */

            int netWidth  = 0; /*<! ����ߴ磬�� */
            int netHeight = 0; /*<! ����ߴ磬�� */

            float mean[3]   = {127.5f, 127.5f, 127.5f};                /*<! ģ��Ԥ���� ����Ԥ����ʽ�� �� */
            float normal[3] = {0.00784314f, 0.00784314f, 0.00784314f}; /*<! ģ��Ԥ���� ����Ԥ����ʽ�� �� */

            float              thresh   = 0.5f;                    /*<! ģ�ͺ��� ��ֵ*/
            int                nClasses = 1;                       /*<! ģ�ͺ��� ���*/
            std::vector<float> strides  = std::vector<float>{1.f}; /*<! ģ�ͺ��� �²�����С*/

            // �㿽�����ݣ�һ���vi��������
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
        // index dataָ�� shapeά��
        typedef std::function<void(int, std::vector<float *>, std::vector<NCHW> &)> CallBack;

        /**
         * @brief ���캯��
         */
        Inference();

        /**
         * @brief ��������
         */
        ~Inference();

        /**
         * @brief ��ʼ������
         * @author tsd
         * @param[in] cfg ���ò���
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int init(const Config &cfg);

        /**
         * @brief ִ������
         * @author tsd
         * @param[in]   inputData       ͼ������
         * @param[out]  outputCallBack  �ص���������Ҫ�Լ�����
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int run(const ImageData &inputData, CallBack &outputCallBack);

        /**
         * @brief ִ�����������ڹؼ���ع�
         * @author tsd
         * @param[in]   inputData   ͼ������
         * @param[out]  vecOutputs  ������ݣ�����ά�ȵ�����
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int run(const ImageData &inputData, std::vector<std::vector<float>> &vecOutputs);

        /**
         * @brief ִ������������һ�������������࣬����ʶ��
         * @author tsd
         * @param[in]   inputData   ͼ������
         * @param[out]  vecOutputs  ������ݣ�����ά�ȵ�����
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int run(const ImageData &inputData, std::vector<float> &vecOutputs);

        /**
         * @brief ִ������������һ���������������࣬������
         * @author tsd
         * @param[in]   inputData   ͼ������
         * @param[out]  score       ����
         * @param[out]  label       ���
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int run(const ImageData &inputData, float &score, int &label);

    private:
        /**
         * @brief ѹ�����ݣ�����ѹ��bgr,rgb, nv12, nv21
         * @author tsd
         * @param[in]   src     ͼ������
         * @param[out]  image   �Ѿ�ѹ��������
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int resizeImage(const ImageData &src, CV::Mat &image);

        /**
         * @brief ת������,�������ͼ��ת����ģ����Ҫ�ĸ�ʽ����Ҫ�ڳ�ʼ����ʱ��ȷ��srcFormat
         * @author tsd
         * @param[in]   src     ͼ������
         * @param[in]   format  ��Ҫת���ɵĸ�ʽ
         * @param[out]  dst     ת���õ�ͼ������
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        void convertImage(const CV::Mat &src, CV::Mat &dst, IMAGE_FORMAT format);

    private:
        /**
         * @brief ��ʼ��ָ��
         * @author tsd
         */
        void create_mnn();
        void create_ncnn();
        void create_rockchip();
        void create_sigmastar();

        /**
         * @brief �ͷ�
         * @author tsd
         */
        void destory();
        void destory_mnn();
        void destory_ncnn();
        void destory_rockchip();
        void destory_sigmastar();

        /**
         * @brief ��ʼ��ģ��
         * @author tsd
         * @param[in] cfg ���ò���
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int init_mnn(const Config &cfg);
        int init_ncnn(const Config &cfg);
        int init_rockchip(const Config &cfg);
        int init_sigmastar(const Config &cfg);

        /**
         * @brief ִ������
         * @author tsd
         * @param[in]   image       ͼ������
         * @param[out]  outputCallBack  �ص���������Ҫ�Լ�����
         * @return
         *      @retval BSJ_AI_FLAG_SUCCESSFUL  (0) �ɹ�
         *      @retval BSJ_AI_FLAG_FAILED      (-2) ʧ��
         */
        int run_mnn(const CV::Mat &image, CallBack &outputCallBack);
        int run_ncnn(const CV::Mat &image, CallBack &outputCallBack);
        int run_rockchip(const CV::Mat &image, CallBack &outputCallBack);
        int run_sigmastar(const CV::Mat &image, CallBack &outputCallBack);

    private:
        std::mutex lock; /*<! ִ���� */

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