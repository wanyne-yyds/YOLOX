#pragma once
#include "Logger.h"
#include "BSJ_AI_config.h"
/*
记录消耗时间到记录文件

V1.0.19.0910 创建                                    by: Yam  2019-09-10
*/
namespace STORE_YAM
{
	class CTimeLogger
	{
	public:
		/*
		函数名     :  CTimeLogger
		原名       :  STORE_YAM::CTimeLogger::CTimeLogger
		描述       :  构造函数，创建消耗时间的记录类
		参数       :  STORE_YAM::CLogger * pLogger,待记录的记录器，已经在用的
		参数       :  bool bLogTime，是否记录文件，方便控制的开关
		访问方式   :  public 
		返回值     :  -
		作者       :  Yam
		日期       :  2019/09/19
		*/
		CTimeLogger(STORE_YAM::CLogger* pLogger, bool bLogTime = false);
		
		/*
		函数名     :  CTimeLogger
		原名       :  STORE_YAM::CTimeLogger::CTimeLogger
		描述       :  构造函数，创建消耗时间的记录类，这里会新创建记录类，只记录时间消耗
		参数       :  const string & sStoreDir，存储目录
		参数       :  const string & sPrefix，前缀
		参数       :  bool bLogTime，是否记录文件，方便控制的开关
		参数       :  STORE_YAM::LOG_FREQ_TYPE_E eFreqType，记录频率类型，参考定义
		参数       :  int nFreqParam，记录频率，参考记录频率类型定义
		参数       :  const std::string & sSuffix，后缀
		访问方式   :  public 
		返回值     :  -
		作者       :  Yam
		日期       :  2019/09/19
		*/
		CTimeLogger(const string& sStoreDir,//存储目录
			const string& sPrefix,//前缀
			bool bLogTime = false,
			STORE_YAM::LOG_FREQ_TYPE_E eFreqType = STORE_YAM::LOG_FREQ_SIZE, int nFreqParam = 100,
			const std::string& sSuffix = ".log"//后缀
		);

		virtual ~CTimeLogger();

	private:
		uint64_t	m_dStartTime;
		uint64_t	m_dEndTime;
		bool	m_bLogTime;
		bool	m_bLoggerCreated;
		STORE_YAM::CLogger* m_pLogger;

	public:
		/*
		函数名     :  SetStart
		原名       :  STORE_YAM::CTimeLogger::SetStart
		描述       :  消耗时间计时开始
		访问方式   :  public 
		返回值     :  void
		作者       :  Yam
		日期       :  2019/09/19
		*/
		void SetStart();


		/*
		函数名     :  LogTime
		原名       :  STORE_YAM::CTimeLogger::LogTime
		描述       :  记录过程的时间消耗
		参数       :  const char * szLogInfo，时间消耗的描述，如：步骤1耗时
		访问方式   :  public 
		返回值     :  void
		作者       :  Yam
		日期       :  2019/09/19
		*/
		void LogTime(const char* szLogInfo);//记录消耗时间
	};
}
