#include "Logger.h"
#include <ctime>
#include "gzCompress.h"
#include<cstring>

//VS环境下，禁用该警告
#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <sys/timeb.h>
#endif

// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>

const double STORE_YAM::CLogger::TIMES_PER_MS = 1.f;
const double STORE_YAM::CLogger::TIMES_PER_S = 1000.f;

STORE_YAM::CLogger::CLogger(void)
	: m_pf(NULL)
	, m_sStoreDir("")
	, m_sExtraName("")
	, m_eLogLevel(STORE_YAM::LOG_LVL_NONE)
	, m_eFreqType(STORE_YAM::LOG_FREQ_NONE)
	, m_nFreqNum(50)
	, m_strExtraInfo("")
	, m_sSuffix(".log")
	, m_bCompress(false)
	, m_eFlushType(LOG_FLUSH_FREQUENCY_NONE)
{
	m_dLogTime = 0.0;
	m_vecBuffer.clear();

	RenameTempFile(m_sStoreDir);
}

STORE_YAM::CLogger::~CLogger(void)
{
	CloseLogger();
}


int STORE_API
STORE_YAM::CLogger::OpenLogger(
	const std::string& sStoreDir, const std::string& sPrefix,
	STORE_YAM::LOG_LEVEL_E eLevel /* = STORE_YAM::LOG_LVL_SIMPLE */, 
	STORE_YAM::LOG_FREQ_TYPE_E eFreqType /* = STORE_YAM::LOG_FREQ_SIZE */, int nFreqParam /* = 100 */, 
	std::string sExtraInfo,
	const std::string& sSuffix,
	bool bCompress /* = false */, 
	STORE_YAM::LOG_FLUSH_TYPE_E eFlushType /* = LOG_FLUSH_FREQ_NONE */)
{
	m_sStoreDir = sStoreDir;
	m_sExtraName = sPrefix;
	m_eLogLevel = eLevel;
	m_eFreqType = eFreqType;
	m_nFreqNum = nFreqParam;
	m_nFlushNum = (m_nFreqNum >> 2);
	m_nFlushCount = 0;
	m_strExtraInfo = sExtraInfo;
	m_bCompress = bCompress;
	m_eFlushType = eFlushType;
	m_sSuffix = sSuffix;

	//自动后缀加.
	if ("" != sSuffix)
	{
		if (sSuffix[0] != '.')
		{
			m_sSuffix = "." + sSuffix;
		}
	}

	if (".~tmp" == m_sSuffix)
	{
		//throw("后缀不能设置为.~tmp");
		return -1;
	}

	RenameTempFile(m_sStoreDir);

	return OpenLog(m_sExtraName);
}


//打开记录层级,sExtraName特殊文件前缀，为空时，自动生成以时间命名的Log.
int  STORE_API
STORE_YAM::CLogger::OpenLog(const std::string& sExtraName)
{
	std::string sTime = STORE_YAM::GetCurTimeString();
	std::string sFullPath = m_sStoreDir + "/";
	m_sExtraName = sExtraName;

	//写入起始时间戳
	m_dLogTime = BSJ_AI::getTickMillitm();
	
	if (NULL != m_pf)
	{
		CloseLogger();
	}

	if ("" == m_sExtraName)
	{
		sFullPath += sTime + m_sSuffix/*".log"*/;
	}
	else
	{
		sFullPath += m_sExtraName + "_" + sTime + m_sSuffix/*".log"*/;
	}
	
	m_sFullPath = sFullPath;

	//生成临时文件
	m_sFullPath += ".~tmp";

	//根据频次设置，记录相应的起始
	if ((STORE_YAM::LOG_FREQ_NUM == m_eFreqType))
	{
		m_nLogsCount = 0;
	}
	
	if (STORE_YAM::LOG_LVL_NONE == m_eLogLevel)
	{
		return 0;
	}

	//打开文件，并记录版本信息
	m_pf = fopen(m_sFullPath.c_str(), "ab+");
	if (NULL == m_pf)
	{
		return  -1;
	}

	if ("" != m_strExtraInfo)
	{
		fprintf(m_pf, "%s\n", m_strExtraInfo.c_str());
	}

	//fprintf(m_pf, "------------------------------------------------Log Begin------------------------------------------------\n");

	//打开后，将缓存存入，防止丢失数据
	for (int i = 0; i < m_vecBuffer.size(); i++)
	{
		//fprintf(m_pf, "%s\n", m_vecBuffer[i].c_str());
		fwrite(m_vecBuffer[i].c_str(), sizeof(char), m_vecBuffer[i].length(), m_pf);
	}

	m_vecBuffer.clear();

	return 0;
}

//记录
int  STORE_API
STORE_YAM::CLogger::Log(STORE_YAM::LOG_LEVEL_E nLevel, const char* strFormat, ...)
{
	if ((m_eLogLevel < nLevel) || (STORE_YAM::LOG_LVL_NONE == m_eLogLevel))
	{
		return 0;
	}

	m_nLogsCount++;

	va_list pArgs;
	va_start(pArgs, strFormat);

#if	defined(_WIN32)
	char szBuffer[1024] = { 0 };
	_vsnprintf(szBuffer, 1024, strFormat, pArgs);
#else
	char szBuffer[1024] = { 0 };
	vsnprintf(szBuffer, 1024, strFormat, pArgs);
#endif

	va_end(pArgs);

	//如果文件没打开，缓存记录
	std::string sBuffer(szBuffer);
	sBuffer += "\n";
	if (!this->IsFileOpened(sBuffer)) 
	{
		return -1;
	}

	/*if (NULL == m_pf)
	{
		m_vecBuffer.push_back(std::string(szBuffer));
		if (m_vecBuffer.size() > BUFFER_NUMBER)
		{
			m_vecBuffer.erase(m_vecBuffer.begin());
		}

		return -1;
	}*/

	fprintf(m_pf, "%s\n", szBuffer);

	bool bReOpen = CheckIsFitFreq();
	if (!bReOpen)
	{
		return -1;
	}

	return 0;
}

int  STORE_API
STORE_YAM::CLogger::Log(STORE_YAM::LOG_LEVEL_E nLevel, bool bWithTimeStamp, const char* strFormat, ...)
{
	if ((m_eLogLevel < nLevel) || (STORE_YAM::LOG_LVL_NONE == m_eLogLevel))
	{
		return 0;
	}

	m_nLogsCount++;

	va_list pArgs;
	va_start(pArgs, strFormat);

#if	defined(_WIN32)
	char szBuffer[1024] = { 0 };
	_vsnprintf(szBuffer, 1024, strFormat, pArgs);
#else
	char szBuffer[1024] = { 0 };
	vsnprintf(szBuffer, 1024, strFormat, pArgs);
#endif

	va_end(pArgs);

	//如果文件没打开，缓存记录
	std::string sBuffer(szBuffer);
	sBuffer += "\n";
	if (!this->IsFileOpened(sBuffer)) 
	{
		return -1;
	}
/*	if (!this->IsFileOpened(std::string(szBuffer))) 
	{
		return -1;
	}*/

	//如果有时间戳标记，应用时间戳
	if (true == bWithTimeStamp)
	{
		std::string sTimeStamp = STORE_YAM::GetCurTimeStringUsec();
		fprintf(m_pf, "[%s]", sTimeStamp.c_str());
	}

	fprintf(m_pf, "%s\n", szBuffer);

	bool bReOpen = CheckIsFitFreq();
	if (!bReOpen)
	{
		return -1;
	}

	return 0;
}

bool STORE_YAM::CLogger::IsFileOpened(const std::string& strBuffer)
{
	if (NULL == m_pf)
	{
		if (m_vecBuffer.size() >= BUFFER_NUMBER)
		{
			m_vecBuffer.erase(m_vecBuffer.begin());
		}
		m_vecBuffer.push_back(strBuffer);

		return false;
	}
	else 
	{
		return true;
	}
}

bool STORE_YAM::CLogger::IsFileOpened(const void* buffer, size_t elementSize, size_t elementCount)
{
	if (NULL == m_pf)
	{
		if (m_vecBuffer.size() >= BUFFER_NUMBER)
		{
			m_vecBuffer.erase(m_vecBuffer.begin());
		}

		std::string strBuffer = "";
		strBuffer.append((char*)buffer, elementSize * elementCount/sizeof(char));
		m_vecBuffer.push_back(strBuffer);

		return false;
	}
	else 
	{
		return true;
	}
}

int STORE_API
STORE_YAM::CLogger::Write(STORE_YAM::LOG_LEVEL_E nLevel, const void* buffer, size_t elementSize, size_t elementCount)
{
	if ((m_eLogLevel < nLevel) || (STORE_YAM::LOG_LVL_NONE == m_eLogLevel))
	{
		return 0;
	}

	m_nLogsCount++;

	//如果文件没打开，缓存记录
	if (!this->IsFileOpened(buffer, elementSize, elementCount)) {
		return -1;
	}

	fwrite(buffer, elementSize, elementCount, m_pf);

	bool bReOpen = CheckIsFitFreq();
	if (!bReOpen)
	{
		return -1;
	}

	return 0;
}

bool
STORE_YAM::CLogger::CheckIsFitFreq()
{
	int nRet = 0;

	if (STORE_YAM::LOG_FREQ_NONE == m_eFreqType)
	{
		return false;
	}
	else if (STORE_YAM::LOG_FREQ_NUM == m_eFreqType)
	{
		if (LOG_FLUSH_FREQUENCY_DYN == m_eFlushType)
		{
			int nNewCount = m_nFlushCount * m_nFlushNum + m_nFlushNum;
			if (m_nLogsCount >= nNewCount)
			{
				m_nFlushCount++;
				fflush(m_pf);
				//Test Code
				//std::cout << "flush[Count=]" << m_nLogsCount << std::endl;
			}
		}
		else if (LOG_FLUSH_FREQUENCY_ATONCE == m_eFlushType)
		{			
			fflush(m_pf);
		}

		//m_nLogsCount++;
		if (m_nLogsCount >= m_nFreqNum)
		{
			m_nFlushCount = 0;
			m_nLogsCount = 0;
			CloseLogger();
			nRet = OpenLog(m_sExtraName);//重新打开
		}
	}
	else if (STORE_YAM::LOG_FREQ_SIZE == m_eFreqType)
	{
		long lSize = ftell(m_pf);
		if (LOG_FLUSH_FREQUENCY_DYN == m_eFlushType)
		{
			int nNewCount = m_nFlushCount * m_nFlushNum + m_nFlushNum;
			nNewCount <<= 10;

			if (lSize >= nNewCount)
			{
				m_nFlushCount++;
				fflush(m_pf);
				//Test Code
				//std::cout << "flush[Size=]" << nNewCount / 1024 << "k" << std::endl;
			}
		}
		else if (LOG_FLUSH_FREQUENCY_ATONCE == m_eFlushType)
		{
			fflush(m_pf);
		}

		if (lSize >= (long)(m_nFreqNum<<10))
		{
			m_nFlushCount = 0;
			CloseLogger();
			nRet = OpenLog(m_sExtraName);//重新打开
		}
	}
	else if (STORE_YAM::LOG_FREQ_TIME == m_eFreqType)
	{
		uint64_t curTime = BSJ_AI::getTickMillitm();
		uint64_t spendTime = curTime - m_dLogTime;
		spendTime /= CLogger::TIMES_PER_S;//计算S时间

		if (LOG_FLUSH_FREQUENCY_DYN == m_eFlushType)
		{
			int nNewCount = m_nFlushCount * m_nFlushNum + m_nFlushNum;
			if (((int)spendTime) >= nNewCount)
			{
				m_nFlushCount++;
				fflush(m_pf);
				//Test Code
				//std::cout << "flush[Time=]" << spendTime << "s" << std::endl;
			}
		}
		else if (LOG_FLUSH_FREQUENCY_ATONCE == m_eFlushType)
		{
			fflush(m_pf);
		}

		if (((int)spendTime) >= m_nFreqNum)
		{
			m_nFlushCount = 0;
			CloseLogger();
			nRet = OpenLog(m_sExtraName);//重新打开
		}
	}

	if (0 != nRet)
	{
		return false;
	}

	return true;
}

int  STORE_API
STORE_YAM::CLogger::CloseLogger()
{
	if (NULL != m_pf)
	{
		fclose(m_pf);
		m_pf = NULL;
	}

	//*.???.~tmp->*.???
	std::string sPath = m_sFullPath.substr(0, m_sFullPath.length() - 5);
	STORE_YAM::RenameFile(m_sFullPath, sPath);

	if (m_bCompress)
	{
		MLM_LNC::Compress Compresser;
		Compresser.CompressFile(sPath.c_str());
	}

	return 0;
}

int STORE_API
STORE_YAM::CLogger::RenameTempFile(const std::string& sTargetDir)
{
	std::vector<string> arrAllTmpFiles;
	int nCount = 0;
	if (!sTargetDir.empty())
	{
		nCount = STORE_YAM::SearchAllFiles(sTargetDir, ".~tmp", arrAllTmpFiles, true);
	}
	std::size_t tPosUL = std::string::npos;
	std::size_t tPosDot = std::string::npos;

	std::string sDir = "", sName = "", sShortName = "", sExt = "";
	std::string sCmpPrefix;
	std::string sCmpSuffix;

	std::string sTime = STORE_YAM::GetCurTimeString();
	int nTimeLen = sTime.length();

	for (int i = 0; i < arrAllTmpFiles.size(); i++)
	{
		/*仅前缀后缀相同的才能够被重命名*/		
		STORE_YAM::ParseFileName(arrAllTmpFiles[i], sDir, sShortName, sName, sExt);

		sCmpPrefix = "";//前缀
		sCmpSuffix = "";//后缀
		tPosUL = sName.find_last_of('_');
		tPosDot = sName.find_first_of(".");

		/*获取前缀*/
		if (std::string::npos != tPosUL)
		{
			sCmpPrefix = sName.substr(0, tPosUL);
		}

		/*获取后缀*/
		if (std::string::npos != tPosDot)
		{
			sCmpSuffix = sName.substr(tPosDot);
		}

		/*前后缀全部相同才进行重命名*/
		if ((sCmpPrefix == m_sExtraName) && (sCmpSuffix == m_sSuffix))
		{
			std::string sPath = arrAllTmpFiles[i].substr(0, arrAllTmpFiles[i].length() - 5);
			STORE_YAM::RenameFile(arrAllTmpFiles[i], sPath);
			if (m_bCompress)
			{
				MLM_LNC::Compress Compresser;
				Compresser.CompressFile(sPath.c_str());
			}
		}
	}

	return nCount;
}
