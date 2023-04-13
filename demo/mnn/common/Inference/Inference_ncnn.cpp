#include "Inference.h"

void BSJ_AI::Inference::create_ncnn() {
}

void BSJ_AI::Inference::destory_ncnn() {
#ifdef USE_NCNN
    m_hNcnn.reset();
#endif
}

#ifdef USE_NCNN
class YoloV5Focus : public ncnn::Layer {
public:
    YoloV5Focus() {
        one_blob_only = true;
    }
    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++) {
            const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }
        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)
#endif // USE_NCNN

int BSJ_AI::Inference::init_ncnn(const Config &cfg) {
#ifdef USE_NCNN
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.model == NULL && cfg.model_path.empty(), "cfg.model == NULL && cfg.model_path.empty()");
    BSJ_AI_MODEL_FORWARD_ASSERT(cfg.param == NULL && cfg.param_path.empty(), "cfg.param == NULL && cfg.param_path.empty()");

    m_hNcnn = std::make_shared<ncnn::Net>();
    m_hNcnn->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    int ret0 = 0;
    int ret1 = 0;

    // 加载模型
    if (cfg.model && cfg.param) {
        ret0 = m_hNcnn->load_param(cfg.param);
        ret1 = m_hNcnn->load_model(cfg.model);
        BSJ_AI_MODEL_FORWARD_ASSERT(cfg.inpNodes.size() != 1 || cfg.oupNodes.empty(), "cfg.inpNodes.size() != 1 || cfg.oupNodes is empty");
    } else {
        ret0 = m_hNcnn->load_param(cfg.param_path.c_str());
        ret1 = m_hNcnn->load_model(cfg.model_path.c_str());
        // 节点
        BSJ_AI_MODEL_FORWARD_ASSERT(cfg.sInpNodes.size() != 1 || cfg.sOupNodes.empty(), "only support cfg.inpNodes.size() != 1 || cfg.sOupNodes is empty");
    }

    BSJ_AI_MODEL_FORWARD_ASSERT(ret0 == -1, "ret0 == -1, load param err");
    BSJ_AI_MODEL_FORWARD_ASSERT(ret1 == -1, "ret1 == -1, load model err");
#else
    LOGE("please use ncnn\n");
    return BSJ_AI_FLAG_FAILED;
#endif // USE_NCNN
    return BSJ_AI_FLAG_SUCCESSFUL;
}

int BSJ_AI::Inference::run_ncnn(const CV::Mat &image, CallBack &outputCallBack) {
#ifdef USE_NCNN
    BSJ_AI_MODEL_FORWARD_ASSERT(image.empty(), "image is empty");
    BSJ_AI_MODEL_FORWARD_ASSERT(m_hNcnn == NULL, "m_hNcnn == NULL");

    ncnn::Mat in;
    switch (image.channels) {
    case 3:
        in = ncnn::Mat::from_pixels((unsigned char *)image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
        break;
    case 1:
        in = ncnn::Mat::from_pixels((unsigned char *)image.data, ncnn::Mat::PIXEL_GRAY, image.cols, image.rows);
        break;
    default:
        LOGE("BSJ_AI::Inference::run_ncnn image channels is not 1 or 3!");
        return -1;
    }

    in.substract_mean_normalize(m_stCfg.mean, m_stCfg.normal);

    int64_t t1 = BSJ_AI::getTickMillitm();
    ncnn::Extractor ex = m_hNcnn->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(m_stCfg.nThread);

    int nodeSize = 0;
    if (m_stCfg.inpNodes.size() == 1) {
        ex.input(m_stCfg.inpNodes[0], in);
        nodeSize = m_stCfg.oupNodes.size();
    } else {
        ex.input(m_stCfg.sInpNodes[0].c_str(), in);
        nodeSize = m_stCfg.sOupNodes.size();
    }

    int nSize = 0;
    for (int i = 0; i < nodeSize; i++) {
        if (m_stCfg.inpNodes.size() == 1) {
            nSize = m_stCfg.oupNodes[i].size();
        } else {
            nSize = m_stCfg.sOupNodes[i].size();
        }

        std::vector<float *> outputs;
        std::vector<ncnn::Mat> ncnnMat;
        ncnnMat.resize(nSize);

        std::vector<NCHW> shapes;
        for (int j = 0; j < nSize; j++) {
            if (m_stCfg.inpNodes.size() == 1) {
                ex.extract(m_stCfg.oupNodes[i][j], ncnnMat[j]);
            } else {
                ex.extract(m_stCfg.sOupNodes[i][j].c_str(), ncnnMat[j]);
            }

            int w = ncnnMat[j].w;
            int h = ncnnMat[j].h;
            int c = ncnnMat[j].c;

            outputs.push_back((float *)ncnnMat[j].data);
            shapes.push_back(NCHW(1, c, h, w));
        }

        outputCallBack(i, outputs, shapes);
    }
    // LOGE("ncnn %lu ms \n", BSJ_AI::getTickMillitm() - t1);
#else
    LOGE("please use ncnn\n");
    return BSJ_AI_FLAG_FAILED;
#endif // USE_NCNN
    return BSJ_AI_FLAG_SUCCESSFUL;
}
