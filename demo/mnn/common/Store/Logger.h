#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
//#include <varargs.h>
using namespace std;
#include "BSJ_AI_config.h"
#include "StorageQuene.h"
/*
文件名: Logger.h
描  述: 记录文件
作  者: Yam
日  期: 2018-12-10
版  本: 18.1210 - Create
*/
/*
历史修改记录：
2018-12-10： V1.0.18.1210，创建文件						        作者：Yam
2018-12-11:                解决64位系统下，记录错误的问题       作者：Yam 
2019-01-21： V1.0.19.0121，增加ms级时间戳						作者：Yam
2019-02-27： V1.0.19.0227，增加存储目录，用于指定存储路径       作者：Yam
2019-03-04： V1.0.19.0304，1、自动创建目录树                    作者：Yam
						   2、按需分割文件                      作者：Yam
						   3、压缩接口                          作者：LNC-李能才
2019-03-14:  V1.0.19.0314, 增加Log缓存，防止数据丢失
2019-03-15： V1.0.19.0315，修改fflush导致的存储速度慢的问题
2019-03-16:  V1.0.19.0316, 简化接口								作者：Yam
2019-03-17:  V1.0.19.0320, 1、修改const
                           2、修改无参构造的默认值
                           3、修改记录等级为LOG_LVL_NONE时，不打开文件
						                                        作者：Yam
2019-05-24:  V1.0.19.0524, 1、BUG:#define LOG_LEVEL_SIMPLE STORE_YAM::LOG_LVL_SIMPLE;
                              改成#define LOG_LEVEL_SIMPLE STORE_YAM::LOG_LVL_SIMPLE
						   2、增加快速操作的宏
																作者：Yam
2019-07-03:  V1.0.19.0703, 1、修改文件存储方式为临时文件方式，防止文件上传被占用
						   2、增加临时文件扫描恢复功能
						   3、增加文件后缀名指定功能
																作者：Yam
2019-07-04:  V1.0.19.0704, 1、控制输入后缀不能是".~tmp"
						   2、输入后缀没有带"."时，自动加上"."
						   3、只重命名前后缀完全一致的文件
																作者：Yam
2019-03-04： V1.0.19.0723，1、修改未指定路径会到默认目录检索                   
						   2、解压缩接口                        作者：LNC-李能才
2019-07-26:  V1.0.19.0726, 1、输入前缀含"."时自动替换成"_"
						   2、输入后缀含"_"时自动替换成"."
						   3、输入后缀是".~tmp"时无法创建文件
						   4、修改前后缀完全一致的条件
																作者：Yam
2019-07-26:  V1.0.19.0727, 1、增加Write接口，可以写入二进制数据
																作者：桂丰	
2019-08-26:  V1.0.19.0826, 1、修改a+为ab+，防止windows写入错误（\n被改写成\r\n）						作者：桂丰
                           2、删除构造函数打开文件       
2019-09-19:  V1.0.19.0919, 1、增加时间消耗统计的记录类			作者：Yam													
*/
namespace STORE_YAM
{
	typedef enum eLoggerFreqType
	{
		LOG_FREQ_NONE = 0,//一直记录，直至关闭
#ifndef LOG_FREQUENCY_NONE
#define LOG_FREQUENCY_NONE STORE_YAM::LOG_FREQ_NONE
#endif
		LOG_FREQ_TIME = 1,//分时记录，按照设定的秒数重新打开记录
#ifndef LOG_FREQUENCY_TIME
#define LOG_FREQUENCY_TIME STORE_YAM::LOG_FREQ_TIME
#endif
		LOG_FREQ_SIZE = 2,//按大小，超过指定的大小（Kb）后重新打开记录
#ifndef LOG_FREQUENCY_SIZE
#define LOG_FREQUENCY_SIZE STORE_YAM::LOG_FREQ_SIZE
#endif
		LOG_FREQ_NUM = 3//按记录条数，超过指定的记录条数后重新打开记录
#ifndef LOG_FREQUENCY_NUM
#define LOG_FREQUENCY_NUM STORE_YAM::LOG_FREQ_NUM
#endif
	}LOG_FREQ_TYPE_E;


	typedef int LOG_LEVEL_E;
	//Log等级，只有高于设置等级的才记录
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_NONE = 0;//不记录
#ifndef LOG_LEVEL_NONE
#define LOG_LEVEL_NONE STORE_YAM::LOG_LVL_NONE
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_SIMPLE = 1;//必要的记录信息
#ifndef LOG_LEVEL_SIMPLE
#define LOG_LEVEL_SIMPLE STORE_YAM::LOG_LVL_SIMPLE
#define LOG_LEVEL_NECESSARY STORE_YAM::LOG_LVL_SIMPLE
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_DEBUG = 2;//含有需要的调试信息
#ifndef LOG_LEVEL_DEBUG
#define LOG_LEVEL_DEBUG STORE_YAM::LOG_LVL_DEBUG
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_DETAIL = 3; //详细信息
#ifndef LOG_LEVEL_DETAIL
#define LOG_LEVEL_DETAIL STORE_YAM::LOG_LVL_DETAIL
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_USR = 4;
#ifndef LOG_LEVEL_USR
#define LOG_LEVEL_USR  STORE_YAM::LOG_LVL_USR
#endif

	//定时刷新数据到存储介质
	typedef int LOG_FLUSH_TYPE_E;

//文件关闭时刷新，平时系统自己刷新
#ifndef LOG_FLUSH_FREQUENCY_NONE
#define LOG_FLUSH_FREQUENCY_NONE 0
#endif
//根据设定的记录文件频率，自动以1/4频率刷新
#ifndef LOG_FLUSH_FREQUENCY_DYN
#define LOG_FLUSH_FREQUENCY_DYN 1
#endif
//所有记录立即刷新，速度会很慢，谨慎使用，除非间隔很大
#ifndef LOG_FLUSH_FREQUENCY_ATONCE
#define LOG_FLUSH_FREQUENCY_ATONCE 2
#endif

	class CLogger
	{
	public:
		CLogger(void);


		virtual ~CLogger(void);

	public:

		/*
		函数名     :  OpenLogger
		原名       :  STORE_YAM::CLogger::OpenLogger
		描述       :  打开记录文件
		参数       :  const string & sStoreDir，存储记录的目录
		              const string & sPrefix，指定记录文件前缀,前缀不能包含标点符号".",否则创建失败
		              STORE_YAM::LOG_LEVEL_E eLevel，记录等级，定义如下：
                          LOG_LVL_NONE                 不记录
						  LOG_LEVEL_SIMPLE(NECESSARY)  必要记录
						  LOG_LEVEL_DEBUG              带调试信息记录
						  LOG_LEVEL_DETAIL             详细记录
						  LOG_LEVEL_USR                带用户特殊信息的记录
		              STORE_YAM::LOG_FREQ_TYPE_E eFreqType，记录文件更新的频率类型
					      LOG_FREQUENCY_NONE           一直记录，不分割文件
						  LOG_FREQUENCY_TIME           按分段时间进行记录，单位是s
						  LOG_FREQUENCY_SIZE           按单位大小进行记录，单位是kb
						  LOG_FREQUENCY_NUM            按行数进行记录
		              int nFreqParam，记录文件更新频率的参数，按类型和单位进行设置
		              string sExtraInfo，特殊记录信息，如版本号
					  string sSuffix,指定文件的后缀名,默认是“.log”,后缀不能包含"_",且不能是".~tmp"
		              bool bCompress，是否对生成的记录文件进行 ★即时压缩
					      true  表示即时压缩
					      false 表示不压缩
		              STORE_YAM::LOG_FLUSH_TYPE_E eFlushType，文件刷新到存储设备的频率
					      LOG_FLUSH_FREQUENCY_NONE     系统自行决定文件刷新到存储介质
						  LOG_FLUSH_FREQUENCY_DYN      按照设置文件更新频率的1/4间隔刷新
						  LOG_FLUSH_FREQUENCY_ATONCE   即时刷新，★非常影响系统速度，若
						        无必要，不建议使用。
		访问方式   :  public 
		返回值     :  int STORE_API
		作者       :  Yam
		日期       :  2019/03/18
		//Example：
		STORE_YAM::CLogger logger;
		//默认参打开记录文件，前缀为空，自动以时间进行命名
		logger.OpenLogger(sRootDir, std::string(""));

		//默认参打开记录文件，前缀为"LDW"，将生成 LDW_(time).log 为名的记录文件。
		logger.OpenLogger(sRootDir, std::string("LDW"));

		//打开记录文件，前缀为"PCW"，将生成 PCW_(time).log 为名的记录文件,等级为只记录必要信息。
		logger.OpenLogger(sRootDir, std::string("PCW"), LOG_LEVEL_SIMPLE);

		//打开记录文件，前缀为"FCW"，将生成 FCW_(time).log 为名的记录文件,等级为只记录必要信息。
		//记录将按照默认的100kb进行分文件记录
		logger.OpenLogger(sRootDir, std::string("FCW"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_SIZE);

		//打开记录文件，前缀为"BD"，将生成 BD_(time).log 为名的记录文件,等级为只记录必要信息。
		//记录将按照5s时间间隔进行分文件记录
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_TIME, 5);

		//打开记录文件，前缀为"BD"，将生成 BD_(time).log 为名的记录文件,等级为只记录必要信息。
		//记录按行数进行分文件记录,每满1000行，将生成新文件。文件首部记录版本的特殊信息
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_NUM, 1000, "V1.0.0.1");

		//打开记录文件，前缀为"BD"，将生成 BD_(time).txt 为名的记录文件,等级为只记录必要信息。
		//记录按行数进行分文件记录,每满1000行，将生成新文件。文件首部记录版本的特殊信息.
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_NUM, 1000, "V1.0.0.2", ".txt");

		//打开记录文件，前缀为"BD"，将生成 BD_(time) 为名的记录文件,等级为只记录必要信息。
		//记录按行数进行分文件记录,每满1000行，将生成新文件。文件首部记录版本的特殊信息.
		//每当文件写完成，文件将被压缩
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_NUM, 1000, "V1.0.0.2", "", true);

		//打开记录文件，前缀为"BD"，将生成 BD_(time).dat 为名的记录文件,等级为只记录必要信息。
		//记录按行数进行分文件记录,每满100K大小，将生成新文件。文件首部记录版本的特殊信息.
		//每当文件写完成，文件将被压缩，每当写完25K（100K/4）数据，文件将会被刷新到存储介质上
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_SIZE, 1000, "V1.0.0.2", ".dat", true, LOG_FLUSH_FREQUENCY_DYN);
		*/
		int STORE_API OpenLogger(
			const string& sStoreDir,
			const string& sPrefix,
			STORE_YAM::LOG_LEVEL_E eLevel = STORE_YAM::LOG_LVL_SIMPLE,
			STORE_YAM::LOG_FREQ_TYPE_E eFreqType = STORE_YAM::LOG_FREQ_SIZE, int nFreqParam = 100,
			string sExtraInfo = "",
			const std::string& sSuffix = ".log",/*后缀*/
			bool bCompress = false,
			STORE_YAM::LOG_FLUSH_TYPE_E eFlushType = LOG_FLUSH_FREQUENCY_NONE);


		/*
        函数名     :  Log  
        原名       :  STORE_YAM::CLogger::Log
        描述       :  写一条带时间戳的记录 
        参数       :  STORE_YAM::LOG_LEVEL_E nLevel，写入等级
                      bool bWithTimeStamp，是否带时间戳
                      const char * strFormat，写入格式
                      ...，待写入的参数列表
        访问方式   :  public
        返回值     :  int,0-成功 其它-失败
        作者       :  Yam
        日期       :  2019/03/04
        */
		int STORE_API Log(STORE_YAM::LOG_LEVEL_E nLevel, bool bWithTimeStamp, const char* strFormat, ...);

		/*
		函数名     :  Write
		原名       :  STORE_YAM::CLogger::Write
		描述       :  写一条二进制数据
		参数       :  STORE_YAM::LOG_LEVEL_E nLevel，写入等级
		参数       :  const void * buffer，待写入的数据指针
		参数       :  size_t elementSize，单个数据结构大小
		参数       :  size_t elementCount，数据结构的数目
		访问方式   :  public 
		返回值     :  int STORE_API， 0-成功， 其它失败
		作者       :  桂丰
		日期       :  2019/07/27
		*/
		int STORE_API Write(STORE_YAM::LOG_LEVEL_E nLevel, const void* buffer, size_t elementSize, size_t elementCount);

		/*
		函数名： CloseLogger
		描  述： 关闭记录文件
		参  数： 无
		返回值：
				 int - 成功0 失败-1
		*/
		int STORE_API CloseLogger();

	private:
		FILE*	m_pf;

		std::string m_sStoreDir;//存储记录文件的路径
		std::string m_sExtraName;
		STORE_YAM::LOG_FREQ_TYPE_E m_eFreqType;
		STORE_YAM::LOG_LEVEL_E m_eLogLevel;
		int			m_nFreqNum;
		double		m_dLogTime;
		int			m_nLogsCount;

		std::string m_strExtraInfo;
		std::string m_sFullPath;
		bool		m_bCompress;

		std::string m_sSuffix;

		//缓存
		std::vector<std::string>	m_vecBuffer;
		const int	BUFFER_NUMBER = 200;//最多缓存200条记录

		STORE_YAM::LOG_FLUSH_TYPE_E m_eFlushType;
		int							m_nFlushNum;
		int							m_nFlushCount;

	public:
		static const double TIMES_PER_MS;
		static const double TIMES_PER_S;

	private:
		bool IsFileOpened(const std::string& strBuffer);
		bool IsFileOpened(const void* buffer, size_t elementSize, size_t elementCount);
		bool CheckIsFitFreq();

		/*
		函数名     :  OpenLog
		原名       :  STORE_YAM::CLogger::OpenLog
		描述       :  打开记录层级,sExtraName特殊文件前缀，为空时，自动生成以时间命名的Log.
		参数       :  const std::string& sExtraName, 特殊前缀
		访问方式   :  public
		返回值     :  int,0-成功 其它-失败
		作者       :  Yam
		日期       :  2019/03/04
		*/
		int STORE_API OpenLog(const std::string& sExtraName);


		/*
		函数名     :  Log
		原名       :  STORE_YAM::CLogger::Log
		描述       :  写一条记录
		参数       :  STORE_YAM::LOG_LEVEL_E nLevel，写入等级
					  const char * strFormat，写入格式
					  ...，待写入的参数列表
		访问方式   :  public
		返回值     :  int,0-成功 其它-失败
		作者       :  Yam
		日期       :  2019/03/04
		*/
		int STORE_API Log(STORE_YAM::LOG_LEVEL_E nLevel, const char* strFormat, ...);

		/*
		函数名     :  RenameTempFile
		原名       :  STORE_YAM::CLogger::RenameTempFile
		描述       :  恢复临时文件
		参数       :  const std::string & sTargetDir，指定目录
		访问方式   :  private 
		返回值     :  int STORE_API，返回恢复成功的个数
		作者       :  Yam
		日期       :  2019/07/03
		*/
		int STORE_API RenameTempFile(const std::string& sTargetDir);

		/*
		函数名     :  GetPrefixAndSuffix
		原名       :  STORE_YAM::CLogger::GetPrefixAndSuffix
		描述       :  根据时间长度，解析前缀和后缀
		参数       :  const std::string & sForSrch，待找前缀和后缀的文件名
		参数       :  int nTimeLen，时间字符串长度
		参数       :  std::string & sPrefix，前缀
		参数       :  std::string & sSuffix，后缀
		访问方式   :  public 
		返回值     :  int STORE_API
		作者       :  Yam
		日期       :  2019/07/05
		*/
		int STORE_API GetPrefixAndSuffix(const std::string& sForSrch, int nTimeLen, std::string &sPrefix, std::string &sSuffix);
	};

#define AD_LOGS(pLogger, sFormat, ...) do{if(NULL!=(pLogger)){(STORE_YAM::CLogger*)(pLogger)->Log(LOG_LEVEL_SIMPLE, false, sFormat, ##__VA_ARGS__);}}while(0)
#define AD_LOGS_T(pLogger, sFormat, ...) do{if(NULL!=(pLogger)){(STORE_YAM::CLogger*)(pLogger)->Log(LOG_LEVEL_SIMPLE, true, sFormat, ##__VA_ARGS__);}}while(0)

#define AD_LOGD(pLogger, sFormat, ...) do{if(NULL!=(pLogger)){(STORE_YAM::CLogger*)(pLogger)->Log(LOG_LEVEL_DEBUG, false, sFormat, ##__VA_ARGS__);}}while(0)
#define AD_LOGD_T(pLogger, sFormat, ...) do{if(NULL!=(pLogger)){(STORE_YAM::CLogger*)(pLogger)->Log(LOG_LEVEL_DEBUG, true, sFormat, ##__VA_ARGS__);}}while(0)

#define AD_LOGF(pLogger, sFormat, ...) do{if(NULL!= (pLogger)){(STORE_YAM::CLogger*)(pLogger)->Log(LOG_LEVEL_DETAIL, false, sFormat, ##__VA_ARGS__);}}while(0)
#define AD_LOGF_T(pLogger, sFormat, ...) do{if(NULL!=(pLogger)){(STORE_YAM::CLogger*)(pLogger)->Log(LOG_LEVEL_DETAIL, true, sFormat, ##__VA_ARGS__);}}while(0)
}
