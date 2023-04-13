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
�ļ���: StorageQuene.h
��  ��: ���л��洢��¼�ļ�
��  ��: Yam
��  ��: 2018-12-10
��  ��: 18.1210 -Create
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
	������     :  SplitString
	ԭ��       :  STORE_YAM::SplitString
	����       :  ���ݷָ����ַ����ָ��ַ�
	����       :  const std::string& sInput,���ָ����ַ���
	              std::vector<std::string> & sSubStrings���洢���ַ���������
	              std::string sSep���ָ�����߷ָ��ַ���
	���ʷ�ʽ   :  public 
	����ֵ     :  int STORE_API�����طָ����ַ�������Ŀ
	����       :  Yam
	����       :  2019/03/16
	*/
	int STORE_API SplitString(const std::string& sInput, std::vector<std::string> &sSubStrings, const std::string& sSep);

	/*
	������     :  CreateDir
	ԭ��       :  STORE_YAM::CreateDir
	����       :  �����༶�򵥼�Ŀ¼����ƽ̨
	����       :  const std::string& sDir,��������Ŀ¼
	���ʷ�ʽ   :  public 
	����ֵ     :  int STORE_API��0-�ɹ� -1 ʧ��
	����       :  Yam
	����       :  2019/03/16
	*/
	int STORE_API CreateDir(const std::string& sDir);

	/*
	�������� GetCurTimeString
	��  ���� ��õ�ǰϵͳʱ����ַ���
	��  ���� ��
	����ֵ��
	         std::string - ϵͳʱ����ַ���
	//Example:
	//    std::string sTime = GetCurTimeString();
	//    std::cout<<sTime<<std::endl;
	*/
	std::string STORE_API GetCurTimeString();

	/*
	�������� GetCurTimeStringUsec
	��  ���� ��õ�ǰϵͳʱ����ַ��������뼶
	��  ���� ��
	����ֵ��
	std::string - ϵͳʱ����ַ���
	//Example:
	//    std::string sTime = GetCurTimeStringUsec();
	//    std::cout<<sTime<<std::endl;
	*/
	std::string STORE_API GetCurTimeStringUsec();//���ms����ʱ�䣬����ʱ���
	std::string STORE_API GetCurTimeStringUsec2();//���ms����ʱ�䣬�����ļ���

	/*
	�������� GetStorageVersion
	��  ���� ��ô洢ϵͳ�汾��
	��  ���� @szVesionBuf - ���ڴ�Ű汾�ŵĻ����������
	         nBufLen - ���ڴ�Ű汾�ŵĻ���������
	����ֵ��
	         int - �汾�ų���
	//Example: 
	//	char szVer[32];
	//	memset(szVer, 0, 32);
	//  STORE_YAM::GetStorageVersion(szVer, 32);
	//	std::cout<<"Version:"<<std::string(szVer)<<std::endl;
	*/
	int STORE_API GetStorageVersion(char* szVesionBuf, int nBufLen);

	/*
	�������� GetStorageVersion
	��  ���� ��ô洢ϵͳ�汾�ŵĶ�Ӧ����
	��  ���� @szDescriptionBuf - ���ڴ�Ű汾�����Ļ����������
	         nBufLen - ���ڴ�Ű汾�ŵĻ���������
	����ֵ��
	         int - �汾��������
	//Example:
	//	char szVer[512];
	//	memset(szVer, 0, 512);
	//  STORE_YAM::GetStorageVersion(szVer, 512);
	//	std::cout<<"Version:"<<std::string(szVer)<<std::endl;
	*/
	int STORE_API GetStorageDescription(char* szDescriptionBuf, int nBufLen);


	/*
	������     :  SearchAllFiles
	ԭ��       :  STORE_YAM::SearchAllFiles
	����       :  ����ĳһĿ¼�µ������ļ������õ�ÿ���ļ�������·��
	����       :  const std::string& sRootDir���������ļ��ĸ�Ŀ¼
	              const std::string& sExtension,�������ļ��ĺ�׺,��  .txt
	              std::vector<std::string> & arrAllFiles�������ļ�·��������
	              bool bSearchBranches���Ƿ������Ŀ¼��־��true��ʾ���Ұ�����Ŀ¼
	���ʷ�ʽ   :  public 
	����ֵ     :  int STORE_API�����ز��ҵ����ļ��� -1ʧ��
	����       :  Yam
	����       :  2019/03/16
	*/
	int STORE_API SearchAllFiles(const std::string& sRootDir, const std::string& sExtension, 
		std::vector<std::string> &arrAllFiles, bool bSearchBranches = false);

	/*
	������     :  SearchAllSubDirs
	ԭ��       :  STORE_YAM::SearchAllSubDirs
	����       :  ����ĳһĿ¼�µ�����Ŀ¼
	����       :  const std::string & sRootDir�������ҵĸ�Ŀ¼
	              std::vector<std::string> & arrAllSubDirs��������ҵ��ĸ�Ŀ¼·��
	              bool bSearchBranches���Ƿ������Ŀ¼��־��true��ʾ���Ұ�����Ŀ¼
	���ʷ�ʽ   :  public 
	����ֵ     :  int STORE_API�����ز��ҵ���Ŀ¼���� -1��ʾʧ��
	����       :  Yam
	����       :  2019/03/16
	*/
	int STORE_API SearchAllSubDirs(const std::string& sRootDir, 
		std::vector<std::string> &arrAllSubDirs, bool bSearchBranches = false);

    /*parse file path and get infos*/
	/*
	������     :  ParseFileName
	ԭ��       :  STORE_YAM::ParseFileName
	����       :  �����ļ�ȫ·�����õ�Ŀ¼�������֣�����չ���������֣����� ��չ��������չ����
	����       :  const std::string & sFullPath���ļ�ȫ·��
	              std::string & sDirectory������ļ����ڵ�Ŀ¼
	              std::string & sShortName������ļ����֣�����չ��
	              std::string & sName������ļ����֣�������չ��
	              std::string & sExtension������ļ���չ��
	���ʷ�ʽ   :  public 
	����ֵ     :  int -1ʧ�� 0-�ɹ�
	����       :  Yam
	����       :  2019/03/16
	*/
	int ParseFileName(const std::string& sFullPath,
                   std::string &sDirectory, std::string &sShortName, std::string &sName, std::string &sExtension);

	int RenameFile(const std::string& sFileSrc, const std::string& sFileDst);

	int ReplaceChar(std::string& sForReplace, char chNeedReplace, char chToReplace);
}
#endif
