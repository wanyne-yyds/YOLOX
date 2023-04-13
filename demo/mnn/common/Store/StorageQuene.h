#ifndef __YAM_STORAGE_H__
#define __YAM_STORAGE_H__

#include <string>
#include <map>
#include <vector>
#ifdef _MSC_VER
#pragma warning(disable:4996)  
#endif
#include <cstdio>
#include <cstdlib>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>

/*
文件名: StorageQuene.h
描  述: 序列化存储记录文件
作  者: Yam
日  期: 2018-12-10
版  本: 18.1210 -Create
*/

#ifndef STORE_API
#define STORE_API
#endif

#ifndef STORE_IN
#define STORE_IN
#endif

#ifndef STORE_OUT
#define STORE_OUT
#endif

#ifndef STORE_INOUT
#define STORE_INOUT
#endif

#ifndef STORE_CALLBACK
#define STORE_CALLBACK
#endif

namespace STORE_YAM
{
	/*
	函数名     :  SplitString
	原名       :  STORE_YAM::SplitString
	描述       :  根据分隔符字符串分割字符
	参数       :  const std::string& sInput,带分隔的字符串
	              std::vector<std::string> & sSubStrings，存储子字符串的数组
	              std::string sSep，分割符或者分割字符串
	访问方式   :  public 
	返回值     :  int STORE_API，返回分割后的字符串的数目
	作者       :  Yam
	日期       :  2019/03/16
	*/
	int STORE_API SplitString(const std::string& sInput, std::vector<std::string> &sSubStrings, const std::string& sSep);

	/*
	函数名     :  CreateDir
	原名       :  STORE_YAM::CreateDir
	描述       :  创建多级或单级目录，跨平台
	参数       :  const std::string& sDir,待创建的目录
	访问方式   :  public 
	返回值     :  int STORE_API，0-成功 -1 失败
	作者       :  Yam
	日期       :  2019/03/16
	*/
	int STORE_API CreateDir(const std::string& sDir);

	/*
	函数名： GetCurTimeString
	描  述： 获得当前系统时间的字符串
	参  数： 无
	返回值：
	         std::string - 系统时间的字符串
	//Example:
	//    std::string sTime = GetCurTimeString();
	//    std::cout<<sTime<<std::endl;
	*/
	std::string STORE_API GetCurTimeString();

	/*
	函数名： GetCurTimeStringUsec
	描  述： 获得当前系统时间的字符串，毫秒级
	参  数： 无
	返回值：
	std::string - 系统时间的字符串
	//Example:
	//    std::string sTime = GetCurTimeStringUsec();
	//    std::cout<<sTime<<std::endl;
	*/
	std::string STORE_API GetCurTimeStringUsec();//获得ms级的时间，用于时间戳
	std::string STORE_API GetCurTimeStringUsec2();//获得ms级的时间，用于文件名

	/*
	函数名： GetStorageVersion
	描  述： 获得存储系统版本号
	参  数： @szVesionBuf - 用于存放版本号的缓存区，输出
	         nBufLen - 用于存放版本号的缓存区长度
	返回值：
	         int - 版本号长度
	//Example: 
	//	char szVer[32];
	//	memset(szVer, 0, 32);
	//  STORE_YAM::GetStorageVersion(szVer, 32);
	//	std::cout<<"Version:"<<std::string(szVer)<<std::endl;
	*/
	int STORE_API GetStorageVersion(char* szVesionBuf, int nBufLen);

	/*
	函数名： GetStorageVersion
	描  述： 获得存储系统版本号的对应描述
	参  数： @szDescriptionBuf - 用于存放版本描述的缓存区，输出
	         nBufLen - 用于存放版本号的缓存区长度
	返回值：
	         int - 版本描述长度
	//Example:
	//	char szVer[512];
	//	memset(szVer, 0, 512);
	//  STORE_YAM::GetStorageVersion(szVer, 512);
	//	std::cout<<"Version:"<<std::string(szVer)<<std::endl;
	*/
	int STORE_API GetStorageDescription(char* szDescriptionBuf, int nBufLen);


	/*
	函数名     :  SearchAllFiles
	原名       :  STORE_YAM::SearchAllFiles
	描述       :  查找某一目录下的所有文件，并得到每个文件的完整路径
	参数       :  const std::string& sRootDir，待查找文件的根目录
	              const std::string& sExtension,待查找文件的后缀,如  .txt
	              std::vector<std::string> & arrAllFiles，保存文件路径的数组
	              bool bSearchBranches，是否查找子目录标志，true表示查找包含子目录
	访问方式   :  public 
	返回值     :  int STORE_API，返回查找到的文件数 -1失败
	作者       :  Yam
	日期       :  2019/03/16
	*/
	int STORE_API SearchAllFiles(const std::string& sRootDir, const std::string& sExtension, 
		std::vector<std::string> &arrAllFiles, bool bSearchBranches = false);

	/*
	函数名     :  SearchAllSubDirs
	原名       :  STORE_YAM::SearchAllSubDirs
	描述       :  查找某一目录下的所有目录
	参数       :  const std::string & sRootDir，待查找的根目录
	              std::vector<std::string> & arrAllSubDirs，保存查找到的根目录路径
	              bool bSearchBranches，是否查找子目录标志，true表示查找包含子目录
	访问方式   :  public 
	返回值     :  int STORE_API，返回查找到的目录数， -1表示失败
	作者       :  Yam
	日期       :  2019/03/16
	*/
	int STORE_API SearchAllSubDirs(const std::string& sRootDir, 
		std::vector<std::string> &arrAllSubDirs, bool bSearchBranches = false);

    /*parse file path and get infos*/
	/*
	函数名     :  ParseFileName
	原名       :  STORE_YAM::ParseFileName
	描述       :  分析文件全路径，得到目录、短名字（带扩展名）、名字（不带 扩展名）、扩展名。
	参数       :  const std::string & sFullPath，文件全路径
	              std::string & sDirectory，输出文件所在的目录
	              std::string & sShortName，输出文件名字，带扩展名
	              std::string & sName，输出文件名字，不带扩展名
	              std::string & sExtension，输出文件扩展名
	访问方式   :  public 
	返回值     :  int -1失败 0-成功
	作者       :  Yam
	日期       :  2019/03/16
	*/
	int ParseFileName(const std::string& sFullPath,
                   std::string &sDirectory, std::string &sShortName, std::string &sName, std::string &sExtension);

	int RenameFile(const std::string& sFileSrc, const std::string& sFileDst);

	int ReplaceChar(std::string& sForReplace, char chNeedReplace, char chToReplace);
}
#endif
