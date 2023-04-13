#ifndef __MYGZCOMPRESS_H__
#define __MYGZCOMPRESS_H__

#include <cstdio>	// for remove()
#include <iostream>

#ifdef _WIN32
#include "zlib.h"
#include "zconf.h"
#else
#include "zlib.h"
#include "zconf.h"
#endif


namespace MLM_LNC
{
	class Compress
	{
	public:
		Compress();
		~Compress();

		int CompressMain(const char * filerAddress);

		//Compress Large File
		int CompressFile(const char* sInputFilePath);
		int CompressFilesBat(const char* sInputDirPath);

		int UnCompressFile(const char * fileAddress, bool bRemove = false);

	private:
		int gzCompress(void *data, size_t ndata, void *zdata, size_t *nzdata);
	};
}


#endif // !__GZCOMPRESS_H__

