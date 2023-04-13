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
�ļ���: Logger.h
��  ��: ��¼�ļ�
��  ��: Yam
��  ��: 2018-12-10
��  ��: 18.1210 - Create
*/
/*
��ʷ�޸ļ�¼��
2018-12-10�� V1.0.18.1210�������ļ�						        ���ߣ�Yam
2018-12-11:                ���64λϵͳ�£���¼���������       ���ߣ�Yam 
2019-01-21�� V1.0.19.0121������ms��ʱ���						���ߣ�Yam
2019-02-27�� V1.0.19.0227�����Ӵ洢Ŀ¼������ָ���洢·��       ���ߣ�Yam
2019-03-04�� V1.0.19.0304��1���Զ�����Ŀ¼��                    ���ߣ�Yam
						   2������ָ��ļ�                      ���ߣ�Yam
						   3��ѹ���ӿ�                          ���ߣ�LNC-���ܲ�
2019-03-14:  V1.0.19.0314, ����Log���棬��ֹ���ݶ�ʧ
2019-03-15�� V1.0.19.0315���޸�fflush���µĴ洢�ٶ���������
2019-03-16:  V1.0.19.0316, �򻯽ӿ�								���ߣ�Yam
2019-03-17:  V1.0.19.0320, 1���޸�const
                           2���޸��޲ι����Ĭ��ֵ
                           3���޸ļ�¼�ȼ�ΪLOG_LVL_NONEʱ�������ļ�
						                                        ���ߣ�Yam
2019-05-24:  V1.0.19.0524, 1��BUG:#define LOG_LEVEL_SIMPLE STORE_YAM::LOG_LVL_SIMPLE;
                              �ĳ�#define LOG_LEVEL_SIMPLE STORE_YAM::LOG_LVL_SIMPLE
						   2�����ӿ��ٲ����ĺ�
																���ߣ�Yam
2019-07-03:  V1.0.19.0703, 1���޸��ļ��洢��ʽΪ��ʱ�ļ���ʽ����ֹ�ļ��ϴ���ռ��
						   2��������ʱ�ļ�ɨ��ָ�����
						   3�������ļ���׺��ָ������
																���ߣ�Yam
2019-07-04:  V1.0.19.0704, 1�����������׺������".~tmp"
						   2�������׺û�д�"."ʱ���Զ�����"."
						   3��ֻ������ǰ��׺��ȫһ�µ��ļ�
																���ߣ�Yam
2019-03-04�� V1.0.19.0723��1���޸�δָ��·���ᵽĬ��Ŀ¼����                   
						   2����ѹ���ӿ�                        ���ߣ�LNC-���ܲ�
2019-07-26:  V1.0.19.0726, 1������ǰ׺��"."ʱ�Զ��滻��"_"
						   2�������׺��"_"ʱ�Զ��滻��"."
						   3�������׺��".~tmp"ʱ�޷������ļ�
						   4���޸�ǰ��׺��ȫһ�µ�����
																���ߣ�Yam
2019-07-26:  V1.0.19.0727, 1������Write�ӿڣ�����д�����������
																���ߣ����	
2019-08-26:  V1.0.19.0826, 1���޸�a+Ϊab+����ֹwindowsд�����\n����д��\r\n��						���ߣ����
                           2��ɾ�����캯�����ļ�       
2019-09-19:  V1.0.19.0919, 1������ʱ������ͳ�Ƶļ�¼��			���ߣ�Yam													
*/
namespace STORE_YAM
{
	typedef enum eLoggerFreqType
	{
		LOG_FREQ_NONE = 0,//һֱ��¼��ֱ���ر�
#ifndef LOG_FREQUENCY_NONE
#define LOG_FREQUENCY_NONE STORE_YAM::LOG_FREQ_NONE
#endif
		LOG_FREQ_TIME = 1,//��ʱ��¼�������趨���������´򿪼�¼
#ifndef LOG_FREQUENCY_TIME
#define LOG_FREQUENCY_TIME STORE_YAM::LOG_FREQ_TIME
#endif
		LOG_FREQ_SIZE = 2,//����С������ָ���Ĵ�С��Kb�������´򿪼�¼
#ifndef LOG_FREQUENCY_SIZE
#define LOG_FREQUENCY_SIZE STORE_YAM::LOG_FREQ_SIZE
#endif
		LOG_FREQ_NUM = 3//����¼����������ָ���ļ�¼���������´򿪼�¼
#ifndef LOG_FREQUENCY_NUM
#define LOG_FREQUENCY_NUM STORE_YAM::LOG_FREQ_NUM
#endif
	}LOG_FREQ_TYPE_E;


	typedef int LOG_LEVEL_E;
	//Log�ȼ���ֻ�и������õȼ��Ĳż�¼
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_NONE = 0;//����¼
#ifndef LOG_LEVEL_NONE
#define LOG_LEVEL_NONE STORE_YAM::LOG_LVL_NONE
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_SIMPLE = 1;//��Ҫ�ļ�¼��Ϣ
#ifndef LOG_LEVEL_SIMPLE
#define LOG_LEVEL_SIMPLE STORE_YAM::LOG_LVL_SIMPLE
#define LOG_LEVEL_NECESSARY STORE_YAM::LOG_LVL_SIMPLE
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_DEBUG = 2;//������Ҫ�ĵ�����Ϣ
#ifndef LOG_LEVEL_DEBUG
#define LOG_LEVEL_DEBUG STORE_YAM::LOG_LVL_DEBUG
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_DETAIL = 3; //��ϸ��Ϣ
#ifndef LOG_LEVEL_DETAIL
#define LOG_LEVEL_DETAIL STORE_YAM::LOG_LVL_DETAIL
#endif
	const STORE_YAM::LOG_LEVEL_E LOG_LVL_USR = 4;
#ifndef LOG_LEVEL_USR
#define LOG_LEVEL_USR  STORE_YAM::LOG_LVL_USR
#endif

	//��ʱˢ�����ݵ��洢����
	typedef int LOG_FLUSH_TYPE_E;

//�ļ��ر�ʱˢ�£�ƽʱϵͳ�Լ�ˢ��
#ifndef LOG_FLUSH_FREQUENCY_NONE
#define LOG_FLUSH_FREQUENCY_NONE 0
#endif
//�����趨�ļ�¼�ļ�Ƶ�ʣ��Զ���1/4Ƶ��ˢ��
#ifndef LOG_FLUSH_FREQUENCY_DYN
#define LOG_FLUSH_FREQUENCY_DYN 1
#endif
//���м�¼����ˢ�£��ٶȻ����������ʹ�ã����Ǽ���ܴ�
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
		������     :  OpenLogger
		ԭ��       :  STORE_YAM::CLogger::OpenLogger
		����       :  �򿪼�¼�ļ�
		����       :  const string & sStoreDir���洢��¼��Ŀ¼
		              const string & sPrefix��ָ����¼�ļ�ǰ׺,ǰ׺���ܰ���������".",���򴴽�ʧ��
		              STORE_YAM::LOG_LEVEL_E eLevel����¼�ȼ����������£�
                          LOG_LVL_NONE                 ����¼
						  LOG_LEVEL_SIMPLE(NECESSARY)  ��Ҫ��¼
						  LOG_LEVEL_DEBUG              ��������Ϣ��¼
						  LOG_LEVEL_DETAIL             ��ϸ��¼
						  LOG_LEVEL_USR                ���û�������Ϣ�ļ�¼
		              STORE_YAM::LOG_FREQ_TYPE_E eFreqType����¼�ļ����µ�Ƶ������
					      LOG_FREQUENCY_NONE           һֱ��¼�����ָ��ļ�
						  LOG_FREQUENCY_TIME           ���ֶ�ʱ����м�¼����λ��s
						  LOG_FREQUENCY_SIZE           ����λ��С���м�¼����λ��kb
						  LOG_FREQUENCY_NUM            ���������м�¼
		              int nFreqParam����¼�ļ�����Ƶ�ʵĲ����������ͺ͵�λ��������
		              string sExtraInfo�������¼��Ϣ����汾��
					  string sSuffix,ָ���ļ��ĺ�׺��,Ĭ���ǡ�.log��,��׺���ܰ���"_",�Ҳ�����".~tmp"
		              bool bCompress���Ƿ�����ɵļ�¼�ļ����� �Ｔʱѹ��
					      true  ��ʾ��ʱѹ��
					      false ��ʾ��ѹ��
		              STORE_YAM::LOG_FLUSH_TYPE_E eFlushType���ļ�ˢ�µ��洢�豸��Ƶ��
					      LOG_FLUSH_FREQUENCY_NONE     ϵͳ���о����ļ�ˢ�µ��洢����
						  LOG_FLUSH_FREQUENCY_DYN      ���������ļ�����Ƶ�ʵ�1/4���ˢ��
						  LOG_FLUSH_FREQUENCY_ATONCE   ��ʱˢ�£���ǳ�Ӱ��ϵͳ�ٶȣ���
						        �ޱ�Ҫ��������ʹ�á�
		���ʷ�ʽ   :  public 
		����ֵ     :  int STORE_API
		����       :  Yam
		����       :  2019/03/18
		//Example��
		STORE_YAM::CLogger logger;
		//Ĭ�ϲδ򿪼�¼�ļ���ǰ׺Ϊ�գ��Զ���ʱ���������
		logger.OpenLogger(sRootDir, std::string(""));

		//Ĭ�ϲδ򿪼�¼�ļ���ǰ׺Ϊ"LDW"�������� LDW_(time).log Ϊ���ļ�¼�ļ���
		logger.OpenLogger(sRootDir, std::string("LDW"));

		//�򿪼�¼�ļ���ǰ׺Ϊ"PCW"�������� PCW_(time).log Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		logger.OpenLogger(sRootDir, std::string("PCW"), LOG_LEVEL_SIMPLE);

		//�򿪼�¼�ļ���ǰ׺Ϊ"FCW"�������� FCW_(time).log Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		//��¼������Ĭ�ϵ�100kb���з��ļ���¼
		logger.OpenLogger(sRootDir, std::string("FCW"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_SIZE);

		//�򿪼�¼�ļ���ǰ׺Ϊ"BD"�������� BD_(time).log Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		//��¼������5sʱ�������з��ļ���¼
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_TIME, 5);

		//�򿪼�¼�ļ���ǰ׺Ϊ"BD"�������� BD_(time).log Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		//��¼���������з��ļ���¼,ÿ��1000�У����������ļ����ļ��ײ���¼�汾��������Ϣ
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_NUM, 1000, "V1.0.0.1");

		//�򿪼�¼�ļ���ǰ׺Ϊ"BD"�������� BD_(time).txt Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		//��¼���������з��ļ���¼,ÿ��1000�У����������ļ����ļ��ײ���¼�汾��������Ϣ.
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_NUM, 1000, "V1.0.0.2", ".txt");

		//�򿪼�¼�ļ���ǰ׺Ϊ"BD"�������� BD_(time) Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		//��¼���������з��ļ���¼,ÿ��1000�У����������ļ����ļ��ײ���¼�汾��������Ϣ.
		//ÿ���ļ�д��ɣ��ļ�����ѹ��
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_NUM, 1000, "V1.0.0.2", "", true);

		//�򿪼�¼�ļ���ǰ׺Ϊ"BD"�������� BD_(time).dat Ϊ���ļ�¼�ļ�,�ȼ�Ϊֻ��¼��Ҫ��Ϣ��
		//��¼���������з��ļ���¼,ÿ��100K��С�����������ļ����ļ��ײ���¼�汾��������Ϣ.
		//ÿ���ļ�д��ɣ��ļ�����ѹ����ÿ��д��25K��100K/4�����ݣ��ļ����ᱻˢ�µ��洢������
		logger.OpenLogger(sRootDir, std::string("BD"), LOG_LEVEL_DEBUG, LOG_FREQUENCY_SIZE, 1000, "V1.0.0.2", ".dat", true, LOG_FLUSH_FREQUENCY_DYN);
		*/
		int STORE_API OpenLogger(
			const string& sStoreDir,
			const string& sPrefix,
			STORE_YAM::LOG_LEVEL_E eLevel = STORE_YAM::LOG_LVL_SIMPLE,
			STORE_YAM::LOG_FREQ_TYPE_E eFreqType = STORE_YAM::LOG_FREQ_SIZE, int nFreqParam = 100,
			string sExtraInfo = "",
			const std::string& sSuffix = ".log",/*��׺*/
			bool bCompress = false,
			STORE_YAM::LOG_FLUSH_TYPE_E eFlushType = LOG_FLUSH_FREQUENCY_NONE);


		/*
        ������     :  Log  
        ԭ��       :  STORE_YAM::CLogger::Log
        ����       :  дһ����ʱ����ļ�¼ 
        ����       :  STORE_YAM::LOG_LEVEL_E nLevel��д��ȼ�
                      bool bWithTimeStamp���Ƿ��ʱ���
                      const char * strFormat��д���ʽ
                      ...����д��Ĳ����б�
        ���ʷ�ʽ   :  public
        ����ֵ     :  int,0-�ɹ� ����-ʧ��
        ����       :  Yam
        ����       :  2019/03/04
        */
		int STORE_API Log(STORE_YAM::LOG_LEVEL_E nLevel, bool bWithTimeStamp, const char* strFormat, ...);

		/*
		������     :  Write
		ԭ��       :  STORE_YAM::CLogger::Write
		����       :  дһ������������
		����       :  STORE_YAM::LOG_LEVEL_E nLevel��д��ȼ�
		����       :  const void * buffer����д�������ָ��
		����       :  size_t elementSize���������ݽṹ��С
		����       :  size_t elementCount�����ݽṹ����Ŀ
		���ʷ�ʽ   :  public 
		����ֵ     :  int STORE_API�� 0-�ɹ��� ����ʧ��
		����       :  ���
		����       :  2019/07/27
		*/
		int STORE_API Write(STORE_YAM::LOG_LEVEL_E nLevel, const void* buffer, size_t elementSize, size_t elementCount);

		/*
		�������� CloseLogger
		��  ���� �رռ�¼�ļ�
		��  ���� ��
		����ֵ��
				 int - �ɹ�0 ʧ��-1
		*/
		int STORE_API CloseLogger();

	private:
		FILE*	m_pf;

		std::string m_sStoreDir;//�洢��¼�ļ���·��
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

		//����
		std::vector<std::string>	m_vecBuffer;
		const int	BUFFER_NUMBER = 200;//��໺��200����¼

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
		������     :  OpenLog
		ԭ��       :  STORE_YAM::CLogger::OpenLog
		����       :  �򿪼�¼�㼶,sExtraName�����ļ�ǰ׺��Ϊ��ʱ���Զ�������ʱ��������Log.
		����       :  const std::string& sExtraName, ����ǰ׺
		���ʷ�ʽ   :  public
		����ֵ     :  int,0-�ɹ� ����-ʧ��
		����       :  Yam
		����       :  2019/03/04
		*/
		int STORE_API OpenLog(const std::string& sExtraName);


		/*
		������     :  Log
		ԭ��       :  STORE_YAM::CLogger::Log
		����       :  дһ����¼
		����       :  STORE_YAM::LOG_LEVEL_E nLevel��д��ȼ�
					  const char * strFormat��д���ʽ
					  ...����д��Ĳ����б�
		���ʷ�ʽ   :  public
		����ֵ     :  int,0-�ɹ� ����-ʧ��
		����       :  Yam
		����       :  2019/03/04
		*/
		int STORE_API Log(STORE_YAM::LOG_LEVEL_E nLevel, const char* strFormat, ...);

		/*
		������     :  RenameTempFile
		ԭ��       :  STORE_YAM::CLogger::RenameTempFile
		����       :  �ָ���ʱ�ļ�
		����       :  const std::string & sTargetDir��ָ��Ŀ¼
		���ʷ�ʽ   :  private 
		����ֵ     :  int STORE_API�����ػָ��ɹ��ĸ���
		����       :  Yam
		����       :  2019/07/03
		*/
		int STORE_API RenameTempFile(const std::string& sTargetDir);

		/*
		������     :  GetPrefixAndSuffix
		ԭ��       :  STORE_YAM::CLogger::GetPrefixAndSuffix
		����       :  ����ʱ�䳤�ȣ�����ǰ׺�ͺ�׺
		����       :  const std::string & sForSrch������ǰ׺�ͺ�׺���ļ���
		����       :  int nTimeLen��ʱ���ַ�������
		����       :  std::string & sPrefix��ǰ׺
		����       :  std::string & sSuffix����׺
		���ʷ�ʽ   :  public 
		����ֵ     :  int STORE_API
		����       :  Yam
		����       :  2019/07/05
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
