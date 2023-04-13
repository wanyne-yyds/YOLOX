#pragma once

#define BSJ_MAX(a, b)              ((a) > (b) ? (a) : (b))
#define BSJ_MIN(a, b)              ((a) < (b) ? (a) : (b))
#define BSJ_ABS(a)                 ((a) > 0 ? (a) : -(a))
#define BSJ_BETWEEN(a, aMin, aMax) BSJ_MIN(BSJ_MAX(a, aMin), aMax)

#ifdef _DEBUG
    #define LOGD(...) (printf(__VA_ARGS__))
    #define LOGI(...) (printf(__VA_ARGS__))
#else
    #define LOGD(...)
    #define LOGI(...)
#endif
#define LOGE(...) (printf(__VA_ARGS__))

#include <stdint.h>
#include <string>
#include <string.h>
#include <vector>
#if defined(_WIN32)
    #include <sys/timeb.h>
#else
    #include <time.h>
    #include <sys/time.h>
#endif

#ifdef _WIN32
    #include <direct.h>
    #include <io.h>
#else
    #include <sys/stat.h>
    #include <unistd.h>
    #include <dirent.h>
#endif

#if defined(_WIN32)
    #include <windows.h>
#else
#endif

namespace BSJ_AI {
    static const char *base64Codes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    static const unsigned char base64Map[256] = {
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 255,
        255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 253, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 62, 255, 255, 255, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 255, 255,
        255, 254, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 255, 255, 255, 255, 255,
        255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255};

    static int base64_encode(const unsigned char *in, int inLen, unsigned char *out) {
        int            i, len2, leven;
        unsigned char *p;
        /* valid output size ? */
        len2  = 4 * ((inLen + 2) / 3);
        p     = out;
        leven = 3 * (inLen / 3);
        for (i = 0; i < leven; i += 3) {
            *p++ = base64Codes[in[0] >> 2];
            *p++ = base64Codes[((in[0] & 3) << 4) + (in[1] >> 4)];
            *p++ = base64Codes[((in[1] & 0xf) << 2) + (in[2] >> 6)];
            *p++ = base64Codes[in[2] & 0x3f];
            in += 3;
        }

        /* Pad it if necessary...  */
        if (i < inLen) {
            unsigned a = in[0];
            unsigned b = (i + 1 < inLen) ? in[1] : 0;
            unsigned c = 0;

            *p++ = base64Codes[a >> 2];
            *p++ = base64Codes[((a & 3) << 4) + (b >> 4)];
            *p++ = (i + 1 < inLen) ? base64Codes[((b & 0xf) << 2) + (c >> 6)] : '=';
            *p++ = '=';
        }

        /* append a NULL byte */
        *p = '\0';

        return (p - out);
    }

    static int base64_decode(const unsigned char *in, int inLen, unsigned char *out) {
        int           t = 0, y = 0, z = 0;
        unsigned char c = 0;
        int           g = 3;

        for (int x = 0; x < inLen; x++) {
            c = base64Map[in[x]];
            if (c == 255) {
                return -1;
            } else if (c == 253) {
                continue;
            } else if (c == 254) {
                c = 0;
                g--;
            }
            t = (t << 6) | c;
            if (++y == 4) {
                out[z++] = (unsigned char)((t >> 16) & 255);
                if (g > 1) {
                    out[z++] = (unsigned char)((t >> 8) & 255);
                }
                if (g > 2) {
                    out[z++] = (unsigned char)(t & 255);
                }
                y = t = 0;
            }
        }
        return z;
    }

    static uint64_t getTickMillitm() {
#if defined(_WIN32)
        struct timeb tb;
        ftime(&tb);

        return (uint64_t)(tb.time * 1000 + tb.millitm);
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return (uint64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
    }

    static bool makeDir(const std::string &strPath) {
#ifdef _WIN32
        if (0 != _access(strPath.c_str(), 2)) {
            if (0 == _mkdir(strPath.c_str())) {
                return true;
            }
        }
#else
        if (0 != access(strPath.c_str(), W_OK | R_OK)) {
            remove(strPath.c_str());
        }

        if (0 == mkdir(strPath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO)) {
            return true;
        }
#endif

        return true;
    }

    static bool createDirectory(const std::string &strPath) {
        char tmpDirPath[1000] = {0};
        for (int i = 0; i < strPath.size(); i++) {
            tmpDirPath[i] = strPath[i];
            if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/') {
#ifdef _WIN32
                makeDir(tmpDirPath);
#else
                if (0 == mkdir(tmpDirPath, S_IRWXU | S_IRWXG | S_IRWXO)) {
                    return true;
                }
#endif
            }
        }
        return true;
    }

    static void sleep(unsigned int millitmSecond) {
#ifdef _WIN32
        Sleep(millitmSecond);
#else
        usleep(millitmSecond * 1000);
#endif
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //	int searchAllFiles(const std::string&, std::vector<std::string>& , const std::string&, bool)		//
    //	功能:																								//
    //		按后缀检索文件。																				//
    //	返回值:																								//
    //		执行结果。																						//
    //			成功:	0																					//
    //			失败：	-1																					//
    //	输入:																								//
    //		sRootDir			检索路径																	//
    //		sExtension			检索后缀名																	//
    //		bSearchBranches		递归检索标识																//
    //	输出:																								//
    //		arrAllFiles			检索文件（绝对路径）														//
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    static int searchAllFiles(const std::string &sRootDir, std::vector<std::string> &arrAllFiles, const std::string &sExtension = "", bool bSearchBranches = false) {
#ifndef _WIN32
        DIR           *dir;
        struct dirent *ptr;
        std::string    sSubDir;
        std::string    sFullName;

        if ((dir = opendir(sRootDir.c_str())) == NULL) {
            return -1;
        }

        while ((ptr = readdir(dir)) != NULL) {
            // if ((DT_REG == ptr->d_type) || (DT_BLK == ptr->d_type) || (DT_UNKNOWN == ptr->d_type))
            if ((DT_REG == ptr->d_type) || (DT_BLK == ptr->d_type)) {
                sFullName = sRootDir + "/" + ptr->d_name;
                if ("" != sExtension) {
                    std::size_t pos = sFullName.find(sExtension, sFullName.length() - sExtension.length() - 1);
                    if (std::string::npos != pos) {
                        arrAllFiles.push_back(sFullName);
                    }
                } else {
                    arrAllFiles.push_back(sFullName);
                }
            } else if (DT_DIR == ptr->d_type) /// dir
            {
                if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
                    continue;
                }

                if (bSearchBranches) {
                    sSubDir = sRootDir;
                    sSubDir += "/";
                    sSubDir += ptr->d_name;
                    searchAllFiles(sSubDir, arrAllFiles, sExtension, bSearchBranches);
                }
            }
        }

        closedir(dir);

        return (int)arrAllFiles.size();
#else
        std::string sSubDir;
        std::string sFullName;
        intptr_t    handle;
        _finddata_t findData;

        sSubDir = sRootDir + "/*.*";
        handle  = _findfirst(sSubDir.c_str(), &findData);
        if (-1 == handle) // 检查是否成功
        {
            return -1;
        }

        do {
            if (_A_SUBDIR == (findData.attrib & _A_SUBDIR)) {
                if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0) {
                    continue;
                }

                if (bSearchBranches) {
                    sSubDir = sRootDir + "/" + findData.name;
                    searchAllFiles(sSubDir, arrAllFiles, sExtension, bSearchBranches);
                }
            } else {
                sFullName = sRootDir + "/" + findData.name;
                if ("" != sExtension) {
                    std::size_t pos = sFullName.find(sExtension, sFullName.length() - sExtension.length() - 1);
                    if (std::string::npos != pos) {
                        arrAllFiles.push_back(sFullName);
                    }
                } else {
                    arrAllFiles.push_back(sFullName);
                }
            }
        } while (_findnext(handle, &findData) == 0);

        _findclose(handle); // 关闭搜索句柄

        return arrAllFiles.size();
#endif

        return 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //	int searchAllFiles(const std::string&, std::vector<std::string>& , std::vector<std::string>&, bool)	//
    //	功能:																								//
    //		按关键字检索文件。																				//
    //	返回值:																								//
    //		执行结果。																						//
    //			成功:	0																					//
    //			失败：	-1																					//
    //	输入:																								//
    //		sRootDir			检索路径																	//
    //		keywords			检索关键字																	//
    //		bSearchBranches		递归检索标识																//
    //	输出:																								//
    //		arrAllFiles			检索文件（绝对路径）														//
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    static int searchAllFiles(const std::string &sRootDir, const std::vector<std::string> &keywords, std::vector<std::string> &arrAllFiles, bool bSearchBranches = false) {
        // search
        int nResult = searchAllFiles(sRootDir, arrAllFiles, "", bSearchBranches);

        // filter
        for (std::vector<std::string>::iterator it = arrAllFiles.begin(); it != arrAllFiles.end();) {
            int pos1 = it->rfind("\\");
            if (pos1 == std::string::npos) {
                pos1 = 0;
            }
            int pos2 = it->rfind("/");
            if (pos2 == std::string::npos) {
                pos2 = 0;
            }
            int pos = BSJ_MAX(pos1, pos2);

            bool bFound = false;
            for (std::vector<std::string>::const_iterator itKeyword = keywords.begin(); itKeyword != keywords.end(); itKeyword++) {
                if (it->find(*itKeyword, pos) != std::string::npos) {
                    bFound = true;
                    break;
                }
            }
            if (bFound) {
                it++;
            } else {
                it = arrAllFiles.erase(it);
            }
        }

        return nResult;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //	void split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimiters = " ")	//
    //	功能:																									//
    //		分割字符串。																							//
    //	输入:																									//
    //		s			字符串																					//
    //		delimiters			分割字符																			//
    //	输出:																									//
    //		tokens			分割数组																				//
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static void split(const std::string &s, std::vector<std::string> &tokens, const std::string &delimiters = " ") {
        // 查找第一个与str中的字符都不匹配的字符，返回它的位置
        std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
        // 查找第一个与str中的字符相匹配的字符，返回它的位置。搜索从pos开始
        std::string::size_type pos = s.find_first_of(delimiters, lastPos);
        while (std::string::npos != pos || std::string::npos != lastPos) { // std::string::npos 还未结束继续执行
            tokens.push_back(s.substr(lastPos, pos - lastPos));
            lastPos = s.find_first_not_of(delimiters, pos);
            pos     = s.find_first_of(delimiters, lastPos);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //	void Dirname(const std::string& s, std::string& p)													//
    //	功能:																								//
    //		去掉文件名，返回目录。																				//
    //	输入:																								//
    //		s			路径																					//
    //	输出:																								//
    //		v			目录																					//
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    static void Dirname(const std::string &s, std::string &p) {
        // 获取最后一个路径
        std::string::size_type iPos = s.find_last_of('/') + 1;
        if (iPos == 0) {
            iPos = s.find_last_of('\\') + 1;
        }
        std::string dirname = s.substr(0, iPos - 1);
        p                   = dirname;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //	void GetName(const std::string& s, std::string& v, bool suffix = false)								//
    //	功能:																								//
    //		返回最后的文件名，可以是目录。																		/
    //	输入:																								//
    //		s			路径																					//
    //		suffix		是否去掉后缀名																			//
    //	输出:																								//
    //		v			文件名																				//
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    static void GetName(const std::string &s, std::string &v, bool suffix = false) {
        // 1.获取不带路径的文件名
        std::string::size_type iPos = s.find_last_of('/') + 1;
        if (iPos == 0) {
            iPos = s.find_last_of('\\') + 1;
        }
        std::string basename = s.substr(iPos, s.length() - iPos);

        // 2. 获取不带后缀的文件名
        if (suffix) {
            basename = basename.substr(0, basename.rfind("."));
        }
        v = basename;
    }

    typedef enum eRoundingType {
        ROUND = 0,
        FLOOR = 1,
        CEIL  = 2
    } ROUNDING_TYPE;

    static int getRounding(int number, int divisor, ROUNDING_TYPE type) {
        int remainder = number % divisor;

        switch (type) {
        case ROUNDING_TYPE::ROUND:
            if (remainder >= (divisor >> 1)) {
                return (number - remainder + divisor);
            } else {
                return (number - remainder);
            }
            break;
        case ROUNDING_TYPE::FLOOR:
            return (number - remainder);
            break;
        case ROUNDING_TYPE::CEIL:
            if (remainder > 0) {
                return (number - remainder + divisor);
            } else {
                return (number - remainder);
            }
            break;
        default:
            return number;
        }
    }
} // namespace BSJ_AI
