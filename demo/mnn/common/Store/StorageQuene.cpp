#include "StorageQuene.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>

#if defined(_WIN32)
#include<io.h>
#include <direct.h>
#include <sys/timeb.h>

#else
#include <unistd.h>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <time.h>
#include <sys/time.h>
#include <dirent.h>
#endif

#ifdef OPENCV_VERSION_HPP

#endif

#ifndef YAM_STORE_VERSION
#define YAM_STORE_VERSION "V1.0.19.0320"
#endif

#ifndef YAM_STORE_DESCRIPTION
#define  YAM_STORE_DESCRIPTION "2019 - 03 - 17:  V1.0.19.0320, \
                               1、修改const \
                               2、修改无参构造的默认值 \
                               3、修改记录等级为LOG_LVL_NONE时，不打开文件"
#endif

#ifndef MAX_PATH
#define MAX_PATH 512
#endif

int 
STORE_YAM::GetStorageVersion(char* szVesionBuf, int nBufLen)
{
	sprintf(szVesionBuf, "%s", YAM_STORE_VERSION);
	return std::strlen(szVesionBuf);
}

int 
STORE_YAM::GetStorageDescription(char* szDescriptionBuf, int nBufLen)
{
	sprintf(szDescriptionBuf, "%s", YAM_STORE_VERSION);
	return std::strlen(szDescriptionBuf);
}

int STORE_YAM::ParseFileName(const std::string& sFullPath,
	std::string &sDirectory, std::string &sShortName, std::string &sName, std::string &sExtension)
{
    std::size_t nPos = sFullPath.find_last_of("/\\");
    if (std::string::npos == nPos)
    {
    	sShortName = sFullPath;
        sDirectory = "";
    }
    else
    {
    	sShortName = sFullPath.substr(nPos + 1);
        sDirectory = sFullPath.substr(0, nPos);
    }

    nPos = sShortName.find_last_of('.');
    if (std::string::npos == nPos)
    {
    	sName = sShortName;
    	sExtension = "";
    }
    else
    {
    	sName = sShortName.substr(0, nPos);
    	sExtension = sShortName.substr(nPos);
    }

    return 0;
}

int STORE_API
STORE_YAM::SearchAllFiles(const std::string& sRootDir, const std::string& sExtension,
	std::vector<std::string> &arrAllFiles, 
	bool bSearchBranches /* = false */)
{
#ifndef _WIN32
	DIR *dir;
	struct dirent *ptr;
	std::string sSubDir;
	std::string sFullName;

	if ((dir = opendir(sRootDir.c_str())) == NULL)
	{
		return -1;
	}

	while ((ptr = readdir(dir)) != NULL)
	{
        if ((DT_REG == ptr->d_type) || (DT_BLK == ptr->d_type))
		{
			sFullName = sRootDir + "/" + ptr->d_name;
			if ("" != sExtension)
			{
				std::size_t pos = sFullName.find(sExtension, sFullName.length() - sExtension.length() - 1);
				if (std::string::npos != pos)
				{
					arrAllFiles.push_back(sFullName);
				}
			}
			else
			{
				arrAllFiles.push_back(sFullName);
			}
			//printf("d_name:%s/%s\n", basePath, ptr->d_name);
		}
		else if (DT_DIR == ptr->d_type)    ///dir
		{
			if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
			{
				continue;
			}

			if (bSearchBranches)
			{
				sSubDir = sRootDir;
				sSubDir += "/";
				sSubDir += ptr->d_name;
				SearchAllFiles(sSubDir, sExtension, arrAllFiles, bSearchBranches);
			}
		}
	}

	closedir(dir);

	return (int)arrAllFiles.size();
#else
	std::string sSubDir;
	std::string sFullName;
	intptr_t handle;
	_finddata_t findData;

	sSubDir = sRootDir + "/*.*";
	handle = _findfirst(sSubDir.c_str(), &findData);
	if (-1 == handle)        // 检查是否成功
	{
		return -1;
	}

	do
	{
		if (_A_SUBDIR == (findData.attrib & _A_SUBDIR))
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
			{
				continue;
			}

			if (bSearchBranches)
			{
				sSubDir = sRootDir + "/" + findData.name;
				SearchAllFiles(sSubDir, sExtension, arrAllFiles, bSearchBranches);
			}
		}
		else
		{
			sFullName = sRootDir + "/" + findData.name;
			if ("" != sExtension)
			{
				std::size_t pos = sFullName.find(sExtension, sFullName.length() - sExtension.length() - 1);
				if (std::string::npos != pos)
				{
					arrAllFiles.push_back(sFullName);
				}
			}
			else
			{
				arrAllFiles.push_back(sFullName);
			}
		}
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);    // 关闭搜索句柄

	return arrAllFiles.size();
#endif

	return 0;
}

int STORE_API
STORE_YAM::SearchAllSubDirs(const std::string& sRootDir,
	std::vector<std::string> &arrAllSubDirs,
	bool bSearchBranches /* = false */)
{
#ifndef _WIN32
	DIR *dir;
	struct dirent *ptr;
	std::string sSubDir;
	std::string sFullName;

	if ((dir = opendir(sRootDir.c_str())) == NULL)
	{
		return -1;
	}

	while ((ptr = readdir(dir)) != NULL)
	{
		if (DT_DIR == ptr->d_type)    ///dir
		{
			if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
			{
				continue;
			}
			sSubDir = sRootDir;
			sSubDir += "/";
			sSubDir += ptr->d_name;

			arrAllSubDirs.push_back(sSubDir);

			if (bSearchBranches)
			{				
				SearchAllSubDirs(sSubDir, arrAllSubDirs, bSearchBranches);
			}
		}
	}

	closedir(dir);

	return (int)arrAllSubDirs.size();
#else
	std::string sSubDir;
	std::string sFullName;
	intptr_t handle;
	_finddata_t findData;

	sSubDir = sRootDir + "/*.*";

	handle = _findfirst(sSubDir.c_str(), &findData);
	if (-1 == handle)        // 检查是否成功
	{
		return -1;
	}

	do
	{
		if (_A_SUBDIR == (findData.attrib & _A_SUBDIR))
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
			{
				continue;
			}

			sSubDir = sRootDir + "/" + findData.name;
			arrAllSubDirs.push_back(sSubDir);

			if (bSearchBranches)
			{
				
				SearchAllSubDirs(sSubDir, arrAllSubDirs, bSearchBranches);
			}
		}
	} while (-1 != _findnext(handle, &findData));

	_findclose(handle);    // 关闭搜索句柄

	return arrAllSubDirs.size();
#endif

	return 0;
}

int
STORE_YAM::CreateDir(const std::string& sDir)
{
	std::string sTotalDir;
	char szCurDir[MAX_PATH];
	if (sDir.size() < 1)
	{
		return -1;
	}

	std::string sTempDir = sDir;
	std::replace(sTempDir.begin(), sTempDir.end(), '\\', '/');

#if defined(_WIN32)
	if (std::string::npos == sTempDir.find(":/"))//如果没有找到盘符开始的字符串
	{
		memset(szCurDir, 0, MAX_PATH);
		getcwd(szCurDir, MAX_PATH);
		strcat(szCurDir, "/");
		strcat(szCurDir, sTempDir.c_str());
		if ('/' != sTempDir[sTempDir.length() - 1])
		{
			strcat(szCurDir, "/");
		}
		sTotalDir = szCurDir;
	}
	else
	{
		sTotalDir = sTempDir;
		if ('/' != sTempDir[sTempDir.length() - 1])
		{
			sTotalDir = sTempDir + "/";
		}
	}

	for (int i = 0; i < sTotalDir.size(); i++)
	{
		if (('/' == sTotalDir[i]) && ('/' != sTotalDir[i + 1]))
		{
			std::string sParent = sTotalDir.substr(0, i);
			if (-1 == _access(sParent.c_str(), 0))
			{
				int nRet = _mkdir(sParent.c_str());     //创建目录
				if (-1 == nRet)
				{
					return -1;
				}
			}
		}
	}

	if (-1 == _access(sTotalDir.c_str(), 0))
	{
		int nRet = _mkdir(sTotalDir.c_str());     //创建目录
		if (-1 == nRet)
		{
			return -1;
		}
	}
#else
	//linux文件系统以’/‘开始
	if ('/' != sTempDir[0])
	{
		memset(szCurDir, 0, MAX_PATH);
		getcwd(szCurDir, MAX_PATH);
		strcat(szCurDir, "/");
		strcat(szCurDir, sTempDir.c_str());
		if ('/' != sTempDir[sTempDir.length() - 1])
		{
			strcat(szCurDir, "/");
		}
		sTotalDir = szCurDir;
	}
	else
	{
		sTotalDir = sTempDir;
		if ('/' != sTempDir[sTempDir.length() - 1])
		{
			sTotalDir = sTempDir + "/";
		}
	}

	for (int i = 0; i < sTotalDir.size() - 1; i++)
	{
		if (('/' == sTotalDir[i]) && ('/' != sTotalDir[i + 1]))
		{
			std::string sParent = sTotalDir.substr(0, i);
			
			if (0 != access(sParent.c_str(), NULL))
			{
				if (-1 == mkdir(sParent.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
				{
					continue;
				}
			}
		}
	}

	if (0 != access(sTotalDir.c_str(), NULL))
	{
		if (-1 == mkdir(sTotalDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
		{
			return -1;
		}
	}
#endif

	return 0;
}

int
STORE_YAM::SplitString(const std::string& sInput, std::vector<std::string> &sSubStrings, const std::string& sSplit)
{
	sSubStrings.clear();

	std::string::size_type pos1, pos2;
	pos2 = sInput.find(sSplit);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		sSubStrings.push_back(sInput.substr(pos1, pos2 - pos1));

		pos1 = pos2 + 1;
		pos2 = sInput.find(sSplit, pos1);
	}

	sSubStrings.push_back(sInput.substr(pos1));

	return sSubStrings.size();
}

std::string
STORE_YAM::GetCurTimeString()
{
	//获取当前时间的字符串
	time_t curtime;
	time(&curtime);
	tm* local; //本地时间 
	local = localtime(&curtime);
	char buf[64] = { 0 };
	strftime(buf, 64, "%Y%m%d%H%M%S", local);
	return std::string(buf);
}

//获取ms级时间字符串
std::string
STORE_YAM::GetCurTimeStringUsec()
{
#if defined(_WIN32)
	//获取当前时间的字符串
	std::string sTimeStamp;
	char buf[64] = { 0 };
	char szMs[8];
	struct timeb tb;
	tm* local; //本地时间 

	memset(buf, 0, 64);
	memset(szMs, 0, 8);

	ftime(&tb);

	
	local = localtime(&tb.time);
	
	strftime(buf, 64, "%Y-%m-%d %H:%M:%S", local);
	sTimeStamp = buf;
	sprintf(szMs, ".%03u", tb.millitm);
	sTimeStamp += szMs;

	return sTimeStamp;

#else
	struct timeval    tv;
	struct timezone tz;
	struct tm         *p;

	gettimeofday(&tv, &tz);
	p = localtime(&tv.tv_sec);
	char buf[64] = { 0 };
	sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d.%06ld", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec);
	return std::string(buf);

#endif
}

std::string
STORE_YAM::GetCurTimeStringUsec2()
{
#if defined(_WIN32)
	//获取当前时间的字符串
	std::string sTimeStamp;
	char buf[64] = { 0 };
	char szMs[8];
	struct timeb tb;
	tm* local; //本地时间 

	memset(buf, 0, 64);
	memset(szMs, 0, 8);

	ftime(&tb);

	
	local = localtime(&tb.time);
	
	strftime(buf, 64, "%Y%m%d%H%M%S", local);
	sTimeStamp = buf;
	sprintf(szMs, "%03u", tb.millitm);
	sTimeStamp += szMs;

	return sTimeStamp;
#else
	struct timeval    tv;
	struct timezone tz;
	struct tm         *p;

	gettimeofday(&tv, &tz);
	p = localtime(&tv.tv_sec);
	char buf[64] = { 0 };
	sprintf(buf, "%04d%02d%02d%02d%02d%02d%06ld", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec);
	return std::string(buf);
#endif
}

int 
STORE_YAM::RenameFile(const std::string& sFileSrc, const std::string& sFileDst)
{
	int nRet = rename(sFileSrc.c_str(), sFileDst.c_str());
	return nRet;
}

int
STORE_YAM::ReplaceChar(std::string& sForReplace, char chNeedReplace, char chToReplace)
{
	int nCount = 0;

	for (unsigned int i = 0; i < sForReplace.length(); i++)
	{
		if (chNeedReplace == sForReplace[i])
		{
			sForReplace[i] = chToReplace;
			nCount++;
		}
	}

	return nCount;
}