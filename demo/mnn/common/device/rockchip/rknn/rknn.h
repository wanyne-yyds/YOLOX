#ifndef _RKNN_H
#define _RKNN_H
#include <stdint.h>
#include <vector>
#include <BSJ_AI_config.h>
#include <BSJ_IMAGE_tool.h>
#include <rknn_api.h>
#include <functional>

//#define RKNN_SUCC                               0       /* execute succeed. */
//#define RKNN_ERR_FAIL                           -1      /* execute failed. */
//#define RKNN_ERR_TIMEOUT                        -2      /* execute timeout. */
//#define RKNN_ERR_DEVICE_UNAVAILABLE             -3      /* device is unavailable. */
//#define RKNN_ERR_MALLOC_FAIL                    -4      /* memory malloc fail. */
//#define RKNN_ERR_PARAM_INVALID                  -5      /* parameter is invalid. */
//#define RKNN_ERR_MODEL_INVALID                  -6      /* model is invalid. */
//#define RKNN_ERR_CTX_INVALID                    -7      /* context is invalid. */
//#define RKNN_ERR_INPUT_INVALID                  -8      /* input is invalid. */
//#define RKNN_ERR_OUTPUT_INVALID                 -9      /* output is invalid. */
//#define RKNN_ERR_DEVICE_UNMATCH                 -10     /* the device is unmatch, please update rknn sdk and npu driver/firmware. */
//#define RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL  -11     /* This RKNN model use pre_compile mode, but not compatible with current driver. */
//#define RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION  -12     /* This RKNN model set optimization level, but not compatible with current driver. */
//#define RKNN_ERR_TARGET_PLATFORM_UNMATCH        -13     /* This RKNN model set target platform, but not compatible with current platform. */


namespace BSJ_AI
{
	typedef struct stNCHW
	{
		int n;
		int c;
		int h;
		int w;

		stNCHW(int _n = 0, int _c = 0, int _h = 0, int _w = 0)
		{
			n = _n; c = _c; h = _h; w = _w;
		}
	}NCHW;


	class RKNN
	{
	public:
		
		RKNN();
		
		~RKNN();
		
		
		//////////////////////////////////////////////////////////////////
		//	int init(int, unsigned char*, uint32_t)						//
		//	功能:														//
		//		加载rknn模型(仅支持单输入模型)。						//
		//	返回值: 													//
		//		执行结果。												//
		//			成功：	RKNN_SUCC									//
		//			失败：	RKNN_ERR_XXX								//
		//	输入:														//
		//		modelSize	模型长度									//
		//		model		模型数据									//
		//		flag		加载选项									//
		//					RKNN_FLAG_PRIOR_HIGH			高优先级	//
		//					RKNN_FLAG_PRIOR_MEDIUM			中优先级	//
		//					RKNN_FLAG_PRIOR_LOW				低优先级	//
		//					RKNN_FLAG_ASYNC_MASK			异步模式	//
		//					RKNN_FLAG_COLLECT_PERF_MASK		可查询模式	//
		//////////////////////////////////////////////////////////////////

		int init(int modelSize, unsigned char* model, uint32_t flag);
		
		
		//////////////////////////////////////////////////////////////////////
		//	int run(const vector<ROI_DATA>&, const std::vector<int>&, 		//
		//			vector<vector<float> >&,								//
		//			std::function< void( int, void*, int ) > )				//
		//	功能:															//
		//		运行rknn模型(仅支持单输入模型)。								//
		//	返回值: 														//
		//		执行结果。													//
		//			成功：	RKNN_SUCC										//
		//			失败：	RKNN_ERR_XXX									//
		//	输入:															//
		//		vecBatchData	输入图像及其ROI								//
		//		call_back	回调函数								//
		//	输出:															//
		//		vecOutputs		输出结果									//
		//////////////////////////////////////////////////////////////////////

		int run(const std::vector<BSJ_AI::ROI_DATA>& vecBatchData, 
				const std::vector<int>& vecCoordsIndex, 
				std::vector<std::vector<float> >& vecOutputs,
				std::function< void( int, void*, int ) > call_back = nullptr,
				bool useRGA = false);


	private:

		int free();

		bool cropImage(const cv::Mat& img, const cv::Rect& r, void* dst_buf, const cv::Size& resize_wh);

		rknn_context	m_hContext;
		rknn_input_output_num m_nIoNumber;
		
		rknn_tensor_attr*	m_aInputAttr;
		rknn_input*		m_aInputData;
		
		rknn_tensor_attr*	m_aOutputAttr;
		rknn_output*	m_aOutputData;

		bool	m_bRunningLock;
		bool	m_bInitLock;
	};
}
#endif