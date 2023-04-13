#pragma once
#include "Logger.h"
#include "BSJ_AI_config.h"
/*
��¼����ʱ�䵽��¼�ļ�

V1.0.19.0910 ����                                    by: Yam  2019-09-10
*/
namespace STORE_YAM
{
	class CTimeLogger
	{
	public:
		/*
		������     :  CTimeLogger
		ԭ��       :  STORE_YAM::CTimeLogger::CTimeLogger
		����       :  ���캯������������ʱ��ļ�¼��
		����       :  STORE_YAM::CLogger * pLogger,����¼�ļ�¼�����Ѿ����õ�
		����       :  bool bLogTime���Ƿ��¼�ļ���������ƵĿ���
		���ʷ�ʽ   :  public 
		����ֵ     :  -
		����       :  Yam
		����       :  2019/09/19
		*/
		CTimeLogger(STORE_YAM::CLogger* pLogger, bool bLogTime = false);
		
		/*
		������     :  CTimeLogger
		ԭ��       :  STORE_YAM::CTimeLogger::CTimeLogger
		����       :  ���캯������������ʱ��ļ�¼�࣬������´�����¼�ֻ࣬��¼ʱ������
		����       :  const string & sStoreDir���洢Ŀ¼
		����       :  const string & sPrefix��ǰ׺
		����       :  bool bLogTime���Ƿ��¼�ļ���������ƵĿ���
		����       :  STORE_YAM::LOG_FREQ_TYPE_E eFreqType����¼Ƶ�����ͣ��ο�����
		����       :  int nFreqParam����¼Ƶ�ʣ��ο���¼Ƶ�����Ͷ���
		����       :  const std::string & sSuffix����׺
		���ʷ�ʽ   :  public 
		����ֵ     :  -
		����       :  Yam
		����       :  2019/09/19
		*/
		CTimeLogger(const string& sStoreDir,//�洢Ŀ¼
			const string& sPrefix,//ǰ׺
			bool bLogTime = false,
			STORE_YAM::LOG_FREQ_TYPE_E eFreqType = STORE_YAM::LOG_FREQ_SIZE, int nFreqParam = 100,
			const std::string& sSuffix = ".log"//��׺
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
		������     :  SetStart
		ԭ��       :  STORE_YAM::CTimeLogger::SetStart
		����       :  ����ʱ���ʱ��ʼ
		���ʷ�ʽ   :  public 
		����ֵ     :  void
		����       :  Yam
		����       :  2019/09/19
		*/
		void SetStart();


		/*
		������     :  LogTime
		ԭ��       :  STORE_YAM::CTimeLogger::LogTime
		����       :  ��¼���̵�ʱ������
		����       :  const char * szLogInfo��ʱ�����ĵ��������磺����1��ʱ
		���ʷ�ʽ   :  public 
		����ֵ     :  void
		����       :  Yam
		����       :  2019/09/19
		*/
		void LogTime(const char* szLogInfo);//��¼����ʱ��
	};
}
