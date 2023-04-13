#include "gzCompress.h"
#include <stdio.h>
#include <stdlib.h>
#include "StorageQuene.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

MLM_LNC::Compress::Compress()
{
}

MLM_LNC::Compress::~Compress()
{
}

int MLM_LNC::Compress::UnCompressFile(const char * fileAddress, bool bRemove)
{
	FILE * fp_out = NULL;
	gzFile inputFile = NULL;
	char buf[1024];
	int re = 0;
	std::string sOutPath = fileAddress;
	sOutPath = sOutPath.substr(0, sOutPath.size() - 3);
	//sOutPath += ".gz";

	if (NULL == (fp_out = fopen(sOutPath.c_str(), "wb")))
	{
		return -1;
	}

	/////////////////////////////////////////////
	inputFile = gzopen(fileAddress, "rb");

	if (NULL == inputFile)
	{
		fclose(fp_out);
		return -1;
	}


	while (NULL == gzeof(inputFile))
	{
		int gz_length = 0;
		gz_length = gzread(inputFile, buf, sizeof(buf));
		// std::cout << "gz_length = " << gz_length << std::endl;
		if (-1 == gz_length)
		{
			re = -1;
			break;
		}
		else
		{
			fwrite(buf, gz_length, 1, fp_out);
			if (ferror(fp_out))
			{
				re = -1;
				break;
			}
		}
		
	}


	

	gzclose(inputFile);

	fclose(fp_out);

	if (-1 == re)
	{
		return re;
	}
	else
	{
		// remove
		if (bRemove)
		{
			remove(fileAddress);
		}
		
	}

	return re;

}

int MLM_LNC::Compress::gzCompress(void *data, size_t ndata, void *zdata, size_t *nzdata)
{
	int ret = -1;
	z_stream c_stream;
	if (!data || !ndata) {
		return -1;
	}
	c_stream.zalloc = NULL;
	c_stream.zfree = NULL;
	c_stream.opaque = NULL;
	if (deflateInit2(&c_stream, Z_BEST_COMPRESSION, Z_DEFLATED, MAX_WBITS + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
		return -1;
	}
	c_stream.next_in = (Bytef *)data;
	c_stream.avail_in = ndata;
	c_stream.next_out = (Bytef *)zdata;
	c_stream.avail_out = *nzdata;
	while (c_stream.avail_in != 0 && c_stream.total_out < *nzdata)
	{
		if (deflate(&c_stream, Z_NO_FLUSH) != Z_OK)
		{
			goto end;
		}
	}
	if (c_stream.avail_in != 0)
	{
		return c_stream.avail_in;
	}
	for (;;)
	{
		ret = deflate(&c_stream, Z_FINISH);
		if (ret == Z_STREAM_END)
		{
			break;
		}
		else if (ret != Z_OK)
		{
			break;
		}
	}
end:
	if (deflateEnd(&c_stream) != Z_OK) {
		return -1;
	}
	*nzdata = c_stream.total_out;
	return 0;

}


int MLM_LNC::Compress::CompressMain(const char * filerAddress)
{
	int err = 0;
	int ret = -1;

	std::string gzFileAddress(filerAddress);
	gzFileAddress = gzFileAddress + ".gz";

	FILE *pfIn = fopen(filerAddress, "rb");
	fseek(pfIn, 0, SEEK_END);
	long nTotalSize = ftell(pfIn);
	if (nTotalSize)
	{
		fseek(pfIn, 0, SEEK_SET);
		char *szBuf = new char[nTotalSize];
		char *szOut = new char[nTotalSize];
		size_t nOutLen = nTotalSize;
		fread(szBuf, 1, nTotalSize, pfIn);
		fclose(pfIn);

		ret = gzCompress(szBuf, nTotalSize, szOut, &nOutLen);

		if (0 == ret) {
			std::cout << "Compress Succeed!\n";

			FILE *pf = fopen(gzFileAddress.c_str(), "wb+");
			fwrite(szOut, 1, nOutLen, pf);
			fclose(pf);

			remove(filerAddress);
		}
		else
		{
			std::cout << "Compress Failed!\n";
			err = -1;
		}

		delete[] szBuf;
		delete[] szOut;
	}
	else
	{
		fclose(pfIn);
		remove(filerAddress);
	}

	return err;
}


int 
MLM_LNC::Compress::CompressFile(const char* sInputFilePath)
{
	FILE * fp_in = NULL; 
	int len = 0; 
	char buf[16384];//16k
	int re = 0;
	std::string sOutPath = sInputFilePath;
	sOutPath += ".gz";

	if (NULL == (fp_in = fopen(sInputFilePath, "rb")))
	{
		return -1;
	}

	/////////////////////////////////////////////
	gzFile out = gzopen(sOutPath.c_str(), "wb6f");

	if (NULL == out)
	{
		fclose(fp_in);
		return -1;
	}

	for (;;)
	{
		len = fread(buf, 1, sizeof(buf), fp_in);

		if (ferror(fp_in))
		{
			re = -1;
			break;
		}

		if (0 == len)
		{
			break;
		}

		if (gzwrite(out, buf, (unsigned)len) != len)
		{
			re = -1;
		}
	}

	gzclose(out);

	fclose(fp_in);

	//If Compress Success, Remove Origin File, else, Remove gz File.
	if (-1 == re)
	{
		re = remove(sOutPath.c_str());
	}
	else
	{
		re = remove(sInputFilePath);
	}

	return re;
}

int 
MLM_LNC::Compress::CompressFilesBat(const char* sInputDirPath)
{
	int nTotal = 0;
	std::vector<std::string> sAllFiles;
	nTotal = STORE_YAM::SearchAllFiles(sInputDirPath, "", sAllFiles);
	if (nTotal < 0)
	{
		return -1;
	}

	nTotal = 0;
	for (int i=0; i<sAllFiles.size(); i++)
	{
		std::size_t nPos = sAllFiles[i].find_last_of('.');
		if (std::string::npos == nPos)
		{
			continue;
		}

		std::string sExt = sAllFiles[i].substr(nPos + 1);
		if (("gz" == sExt) || (".gz" == sExt))
		{
			continue;
		}
		else
		{
			int nRet = CompressFile(sAllFiles[i].c_str());
			if (0 == nRet)
			{
				nTotal++;
			}
		}
	}

	return nTotal;
}