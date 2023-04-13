#include "Mat.h"

namespace BSJ_AI {
namespace CV {
int Laplacian(const Mat& src, Mat& dst, int ksize, double scale, double delta, int borderType) {
	if (src.empty()) {
		return BSJ_AI_FLAG_BAD_PARAMETER;
	}

	if (ksize != 1 && ksize != 3) {
		LOGE("BSJ_AI::CV::Laplacian ksize == 3 ||  ksize == 1\n");
		return BSJ_AI_FLAG_BAD_PARAMETER;
	}
	dst.release();
	dst = Mat(src.rows - 2, src.cols - 2, 1);
	unsigned char* ptr = (unsigned char*)dst.data;
		
	Mat2f kernel(3, 3, 1);
	float K[2][9] = { { 0, 1, 0, 1, -4, 1, 0, 1, 0 }, { 2, 0, 2, 0, -8, 0, 2, 0, 2 } };
	::memcpy(kernel.data, K[ksize == 3], sizeof(float) * kernel.total());
		

	for (int row = 1; row < src.rows - 1; row++) {
		unsigned char* ptr0 = (unsigned char*)(src.data + (row - 1) * src.cols);
		unsigned char* ptr1 = (unsigned char*)(src.data + row * src.cols);
		unsigned char* ptr2 = (unsigned char*)(src.data + (row + 1) * src.cols);
		for (int col = 1; col < src.cols -1; col++) {
			float value = ptr0[col - 1] * kernel.data[0]  + ptr0[col] * kernel.data[1] + ptr0[col + 1] * kernel.data[2] + 
				ptr1[col - 1] * kernel.data[3] + ptr1[col] * kernel.data[4] + ptr1[col + 1] * kernel.data[5] + 
				ptr2[col - 1] * kernel.data[6] + ptr2[col] * kernel.data[7] + ptr2[col + 1] * kernel.data[8];
			*ptr++ = (unsigned char)BSJ_BETWEEN(value, 0, 255);
		}
	}

	return BSJ_AI_FLAG_SUCCESSFUL;
}

}
}

