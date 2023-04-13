#include "TimeLogger.h"


STORE_YAM::CTimeLogger::CTimeLogger(STORE_YAM::CLogger* pLogger, bool bLogTime)
	: m_bLogTime(bLogTime)
	, m_dStartTime(0.0)
	, m_dEndTime(0.0)
	, m_pLogger(NULL)
	, m_bLoggerCreated(false)
{
	m_pLogger = pLogger;
}

STORE_YAM::CTimeLogger::CTimeLogger(const string& sStoreDir,//�洢Ŀ¼
	const string& sPrefix,//ǰ׺
	bool bLogTime,
	STORE_YAM::LOG_FREQ_TYPE_E eFreqType, int nFreqParam,
	const std::string& sSuffix
)
	: m_bLogTime(bLogTime)
	, m_dStartTime(0.0)
	, m_dEndTime(0.0)
	, m_pLogger(NULL)
	, m_bLoggerCreated(false)
{

	m_pLogger = new STORE_YAM::CLogger;
	if (NULL != m_pLogger)
	{
		m_bLoggerCreated = true;
	}

	int nRet = 0;
	nRet = m_pLogger->OpenLogger(sStoreDir, sPrefix, STORE_YAM::LOG_LVL_SIMPLE, 
		eFreqType, nFreqParam, 
		"",
		sSuffix);
	if (0 != nRet)
	{
		delete m_pLogger;
		m_pLogger = NULL;
	}
}

STORE_YAM::CTimeLogger::~CTimeLogger()
{
	if (m_bLoggerCreated)
	{
		m_pLogger->CloseLogger();
		delete m_pLogger;
		m_pLogger = NULL;
	}
}

void 
STORE_YAM::CTimeLogger::SetStart()
{
	if (false == m_bLogTime)
	{
		return;
	}

	if (NULL == m_pLogger)
	{
		return;
	}

	m_dStartTime = BSJ_AI::getTickMillitm();
	m_dEndTime = m_dStartTime;
}

void 
STORE_YAM::CTimeLogger::LogTime(const char* szLogInfo)
{
	if (false == m_bLogTime)
	{
		return;
	}

	if (NULL == m_pLogger)
	{
		return;
	}

	m_dEndTime = BSJ_AI::getTickMillitm();

	AD_LOGS(m_pLogger, 
		"%s spend: %llu ms",
		szLogInfo, 
		(m_dEndTime - m_dStartTime));
}