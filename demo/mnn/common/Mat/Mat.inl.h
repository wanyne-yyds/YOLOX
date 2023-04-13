#pragma once
#include "Mat.h"
namespace BSJ_AI {
namespace CV {
///////////////////// Mat /////////////////////////////////////
template<typename T>
Mat_<T>::Mat_(int _rows, int _cols, int _channels, T* _data) {
    rows = 0;
    cols = 0;
    channels = 0;
    refcount = 0;
    data = 0;

    this->init(_rows, _cols, _channels, _data);
}

template<typename T>
Mat_<T>::Mat_(const Size& size, int _channels, T* _data) {
    rows = 0;
    cols = 0;
    channels = 0;
    refcount = 0;
    data = 0;

    this->init(size.height, size.width, _channels, _data);
}

template<typename T>
Mat_<T>::Mat_(const Mat_<T>& A) {
    rows = 0;
    cols = 0;
    channels = 0;
    refcount = 0;
    data = 0;

   this->init(A.rows, A.cols, A.channels, A.data);
   refcount = A.refcount;

   if (refcount) {
       cv_xadd(refcount, 1);
   }
}

// 初始化
template<typename T>
void Mat_<T>::init(int _rows, int _cols, int _channels, T* _data) {
    if (_rows <= 0 || _cols <= 0 || _channels <= 0) {
        return;
    }

    if (_data == 0) {
        if (refcount && this->data) {
            if (this->total() < _rows * _cols * _channels) {
                return;
            } else {
                this->rows = _rows;
                this->cols = _cols;
                this->channels = _channels;
            }
        } else {
            this->release();
            this->rows = _rows;
            this->cols = _cols;
            this->channels = _channels;
            this->data = (T*)fastMalloc(this->total() * sizeof(T));
            refcount = new int[1];
            *refcount = 1;
            
        }
    } else {
        this->release();
        this->rows = _rows;
        this->cols = _cols;
        this->channels = _channels;
        this->data = _data;
    }
    return;
}

template<typename T>
Mat_<T>::~Mat_() {
    this->release();
}

template<typename T>
void Mat_<T>::release() {
    if (refcount && cv_xadd(refcount, -1) == 1) {
        fastFree(data);
        delete[] refcount;
    }

    data = 0;
    rows = 0;
    cols = 0;
    channels = 0;
    refcount = 0;
}

template<typename T>
bool Mat_<T>::empty() const {
    return data == 0 || total() == 0;
}

template<typename T>
bool Mat_<T>::setZeros() {
    if (this->data) {
        memset(this->data, 0x00, rows * cols * channels * sizeof(T));
        return true;
    }
    else {
        return false;
    }
}

template<typename T>
bool Mat_<T>::fill(T value) {
    if (!this->data) {
        return false;
    }
    int size = (int)total();
    T* ptr = (T*)data;

    for (; size > 0; size--) {
        *ptr++ = value;
    }
    return true;
}

template<typename T>
bool Mat_<T>::fill(T value, int _start, int _end) {
    if (!this->data) {
        return false;
    }
    int size = (int)total();
    if (_start < 0 && _end > size && _start > _end) {
        return false;
    }

    for (int i = _start; i < _end; i++) {
        this->data[i] = (T)value;
    }

    return true;
}

template<typename T>
bool Mat_<T>::fill(Scalar _value, Rect roi) {

    if (!this->data) {
        return false;
    }
    int size = (int)total();
    if (roi.x < 0 && roi.y < 0 && roi.x + roi.width > cols && roi.y + roi.height > rows) {
        return false;
    }


    for (int i = roi.y; i < (roi.y + roi.height); i++) {
        T* ptr = (T*)this->data + (i * cols + roi.x) * channels;
        for (int j = roi.x; j < (roi.x + roi.width); j++) {
            for (int c = 0; c < channels; c++) {
                ptr[c] = (T)_value.val[c];
            }
            ptr += channels;
        }
    }

    return true;
}

template<typename T>
size_t Mat_<T>::total() const {
    return rows * cols * channels;
}

template<typename T>
bool Mat_<T>::setEye() {
    if (this->data) {
        memset(this->data, 0x00, rows * cols * channels * sizeof(T));

        for (int i = 0; i < rows; i++) {
            Rect roi = Rect(i, i, 1, 1);
            fill(Scalar_<T>(1, 1, 1, 1), roi);
        }

        return true;
    }
    else {
        return false;
    }
}

template<typename T>
bool Mat_<T>::det(T& determinant) const {
    if (this->channels != 1 || this->data == 0 || this->cols * this->rows == 0 || this->cols != this->rows) {
        return false;
    }

    switch (this->cols) {
    case 1:
        determinant = this->data[0];
        return true;
    case 2:
        determinant = data[0] * data[3] - data[1] * data[2];
        return true;
    case 3:
        determinant =
            data[0] * (data[4] * data[8] - data[5] * data[7])
            - data[1] * (data[3] * data[8] - data[5] * data[6])
            + data[2] * (data[3] * data[7] - data[4] * data[6]);
        return true;
    }

    return false;
}

// 交换 r1 r2行
template<typename T>
bool Mat_<T>::SwapRow(int r1, int r2) {
    if (r1 == r2) {
        return false;// r1 r2 相等，输入错误
    }
    // 先备份r1数据
    T* tmp = new T[this->cols];
    memcpy(tmp, (T*)(this->data + r1 * this->cols), this->cols * sizeof(T));

    // 把r2换到r1
    memcpy((T*)(this->data + r1 * this->cols), (T*)(this->data + r2 * this->cols), this->cols * sizeof(T));

    // 把tmp换给r2
    memcpy((T*)(this->data + r2 * this->cols), tmp, this->cols * sizeof(T));

    delete[] tmp;
    return true;
}

// 第r行乘scalar
template<typename T>
bool Mat_<T>::ScaleRow(int r, T scalar) {
    T abs_scalar = BSJ_ABS(scalar);
    if (abs_scalar < 1.0e-8) {
        return false; // 非法输入
    }
    for (int i = 0; i < this->cols; i++) {
        this->data[r * this->cols + i] *= scalar;
    }
    return true;
}

// inv
template<typename T>
Mat_<T> Mat_<T>::inv() const {
    if (this->cols > 3 && this->channels == 1) {
        Mat_<T> matInp = Mat_<T>(this->rows, this->cols, this->channels);
        ::memcpy(matInp.data, this->data, this->rows * this->cols * this->channels * sizeof(T));
        Mat_<T> matInv(this->rows, this->cols, 1);
        matInv.setEye();

        /* Convert input to the identity matrix via elementary row operations.
           The ith pass through this loop turns the element at i,i to a 1
           and turns all other elements in column i to a 0. */
        for (int i = 0; i < matInp.rows; i++) {
            // 判断当前行的i位置是否为0
            if (BSJ_ABS(matInp.data[i * matInp.cols + i]) < 1e-127) {
                /* We must swap rows to get a nonzero diagonal element. */
                int r = 0;
                for (r = i + 1; r < matInp.rows; ++r) {
                    // if (input.data[r*input.col + i] != 0.0) 
                    if (BSJ_ABS(this->data[r * matInp.cols + i]) > 1e-127) break;
                }
                /* Every remaining element in this column is zero, so this
                       matrix cannot be inverted.
                */
                //奇异矩阵
                if (r == matInp.rows) return Mat_<T>();
                // 交换 某一行
                matInp.SwapRow(i, r);
                matInv.SwapRow(i, r);

            }

            /* Scale this row to ensure a 1 along the diagonal.
               We might need to worry about overflow from a huge scalar here.
            */
            // if (ABS(input.data[i*input.col + i]) < 1e-8)
            if (BSJ_ABS(matInp.data[i * matInp.cols + i]) < 1e-127) return Mat_<T>();;
            T scalar = (T)1.0 / matInp.data[i * matInp.cols + i];

            // 除以scalar
            matInp.ScaleRow(i, scalar);
            matInv.ScaleRow(i, scalar);

            /* Zero out the other elements in this column. */
            for (int j = 0; j < matInp.rows; j++) {
                if (i == j) continue;

                T shear_needed = -matInp.data[j * matInp.cols + i];

                matInp.ShearRow(j, i, shear_needed);
                matInv.ShearRow(j, i, shear_needed);
            }
        }

        return matInv;
    }

    // det
    T determinant = 0;
    bool bResult = this->det(determinant);
    if (!bResult || determinant == 0) {
        return Mat_<T>(0, 0, 0);
    }
    T coeff = 1 / determinant;

    switch (this->cols) {
    case 1: {
        Mat_<T> matInv(1, 1, 1);
        matInv.data[0] = 1 * coeff;
        return matInv;
    }
    case 2: {
        Mat_<T> matInv(2, 2, 1);
        matInv.data[0] = data[3] * coeff;
        matInv.data[1] = -data[1] * coeff;
        matInv.data[2] = -data[2] * coeff;
        matInv.data[3] = data[0] * coeff;
        return matInv;
    }
    case 3: {
        Mat_<T> matInv(3, 3, 1);
        matInv.data[0] = (data[4] * data[8] - data[5] * data[7]) * coeff;
        matInv.data[3] = -(data[3] * data[8] - data[5] * data[6]) * coeff;
        matInv.data[6] = (data[3] * data[7] - data[4] * data[6]) * coeff;
        matInv.data[1] = -(data[1] * data[8] - data[2] * data[7]) * coeff;
        matInv.data[4] = (data[0] * data[8] - data[2] * data[6]) * coeff;
        matInv.data[7] = -(data[0] * data[7] - data[1] * data[6]) * coeff;
        matInv.data[2] = (data[1] * data[5] - data[2] * data[4]) * coeff;
        matInv.data[5] = -(data[0] * data[5] - data[2] * data[3]) * coeff;
        matInv.data[8] = (data[0] * data[4] - data[1] * data[3]) * coeff;
        return matInv;
    }
    }

    return Mat_<T>(0, 0, 0);
}

// transpose
template<typename T>
Mat_<T> Mat_<T>::t() const {
    Mat_<T> transpose(this->cols, this->rows, this->channels);

    for (int r = 0; r < this->rows; r++) {
        for (int c = 0; c < this->cols; c++) {
            for (int ch = 0; ch < this->channels; ch++) {
                transpose.data[(c * rows + r) * channels + ch] = this->data[(r * cols + c) * channels + ch];
            }
        }
    }

    return transpose;
}

// a * aT
template<typename T>
Mat_<T> Mat_<T>::ATA() const {
    if (channels != 1) {
        return Mat_<T>(0, 0, 0);
    }

    Mat_<T> ata(this->cols, this->cols, 1);

    for (int c1 = 0; c1 < this->cols; c1++) {
        for (int c2 = c1; c2 < this->cols; c2++) {
            T value = 0;
            for (int r = 0; r < this->rows; r++) {
                value += (data[r * cols + c1] * data[r * cols + c2]);
            }
            ata.data[c1 * cols + c2] = value;
            ata.data[c2 * cols + c1] = value;
        }
    }

    return ata;
}

// operator =
template<typename T>
Mat_<T> Mat_<T>::operator=(const Mat_<T> &A) {
    if (this == &A) {
        return *this;
    }

    this->init(A.rows, A.cols, A.channels, A.data);
    if (A.refcount) {
        cv_xadd(A.refcount, 1);
    }

    refcount = A.refcount;
    return *this;
}

// operator *
template<typename T>
Mat_<T> Mat_<T>::operator*(const Mat_<T>& B) {
    if (this->channels != 1 || B.channels != 1 || this->data == 0 || B.data == 0 || B.rows * B.cols * this->rows * this->cols == 0 || B.rows != this->cols) {
        return Mat_<T>(0, 0, 0);
    }

    Mat_<T> AB(this->rows, B.cols, 1);

    for (int r = 0; r < AB.rows; r++) {
        for (int c = 0; c < AB.cols; c++) {
            int posA = r * this->cols;
            int posB = c;
            int posAB = r * AB.cols + c;
            AB.data[posAB] = 0;
            for (int k = 0; k < this->cols; k++) {
                AB.data[posAB] += (this->data[posA] * B.data[posB]);
                posA++;
                posB += B.cols;
            }
        }
    }

    return AB;
}

// operator *=
template<typename T>
Mat_<T> Mat_<T>::operator*=(const T& b) {
    int size = total();
    for (int i = 0; i < size; i++) {
        this->data[i] *= b;
    }
    return *this;
}

// operator /=
template<typename T>
Mat_<T> Mat_<T>::operator/=(const T& b) {
    if (!b) {
        return Mat_<T>(0, 0, 0);
    }
    int size = total();
    for (int i = 0; i < size; i++) {
        this->data[i] *= b;
    }
    return *this;
}

// operator +=
template<typename T>
Mat_<T> Mat_<T>::operator+=(const Mat_<T>& A) {
    int size = this->total();
    if (!this->data || !A.data || !size || !size != A.total()) {
        return Mat_<T>(0, 0, 0);
    }

    Mat_<T> B(A);

    for (int i = 0; i < size; i++) {
        B.data[i] = this->data[i] + A.data[i];
    }
    return B;
}

// operator -=
template<typename T>
Mat_<T> Mat_<T>::operator-=(const Mat_<T>& A) {
    int size = this->total();
    if (!this->data || !A.data || !size || !size != A.total()) {
        return Mat_<T>(0, 0, 0);
    }

    Mat_<T> B(A);

    for (int i = 0; i < size; i++) {
        B.data[i] = this->data[i] - A.data[i];
    }
    return B;
}

template<typename T>
Mat_<T> Mat_<T>::operator()(const Rect& roi) const {
    if (this->empty() || !roi.area()) {
        return Mat_<T>(0, 0, 0);
    }
    Mat_<T> A(roi.height, roi.width, this->channels);
    T* dst = (T*)A.data;

    int bottom = roi.y + roi.height;
    int lenght = roi.width * this->channels;

    for (int i = roi.y; i < bottom; i++) {
        T* src = (T*)this->data + (i * this->cols + roi.x) * this->channels;
        memcpy(dst, src, lenght * sizeof(T));
        dst += lenght;
    }

    return A;
}


// mean
template<typename T>
Mat_<T> Mat_<T>::mean(int axis) const {
    if (!this->total()) {
        return Mat_<T>();
    }

    if (axis == 0) {
        // 按c的一行
        Mat_<T> A(1, cols, channels);
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < cols; i++) {
                float sumAll = 0.;
                for (int j = 0; j < rows; j++) {
                    int index = (j * cols + i) * channels + c;
                    sumAll += this->data[index];
                }
                A.data[c * cols + i] = T(sumAll / rows);
            }
        }
        return A;
    }
    else if (axis == 1) {
        // 按c的一列
        Mat_<T> A(rows, 1, channels);
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < rows; i++) {
                float sumAll = 0.;
                for (int j = 0; j < cols; j++) {
                    int index = (i * cols + j) * channels + c;
                    sumAll += this->data[index];
                }
                A.data[c * rows + i] = T(sumAll / cols);
            }
        }
        return A;
    }
    else if (axis == 2) {
        // 按c的一面
        Mat_<T> A(1, 1, channels);
        for (int c = 0; c < channels; c++) {
            float sumAll = 0.;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int index = (i * cols + j) * channels + c;
                    sumAll += this->data[index];
                }
            }
            A.data[c] = T(sumAll / (rows * cols));
        }
        return A;
    }
    else {
        // 全部
        Mat_<T> A(1, 1, 1);
        float sumAll = 0.;
        int length = total();
        for (int i = 0; i < length; i++) {
            sumAll += this->data[i];
        }
        A.data[0] = T(sumAll / length);
        return A;
    }

    return Mat_<T>();
}


// sum
template<typename T>
Mat_<T> Mat_<T>::sum(int axis) const {
    if (!this->total()) {
        return Mat_<T>();
    }

    // rows
    if (axis == 0) {
        Mat_<T> A(1, cols, channels);
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < cols; i++) {
                float sumAll = 0.;
                for (int j = 0; j < rows; j++) {
                    int index = (j * cols + i) * channels + c;
                    sumAll += this->data[index];
                }
                A.data[c * cols + i] = T(sumAll);
            }
        }
        return A;
    }
    else if (axis == 1) {
        // cols
        Mat_<T> A(rows, 1, channels);
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < rows; i++) {
                float sumAll = 0.;
                for (int j = 0; j < cols; j++) {
                    int index = (i * cols + j) * channels + c;
                    sumAll += this->data[index];
                }
                A.data[c * rows + i] = T(sumAll);
            }
        }
        return A;
    }
    else if (axis == 2) {
        // channel
        Mat_<T> A(1, 1, channels);
        for (int c = 0; c < channels; c++) {
            float sumAll = 0.;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int index = (i * cols + j) * channels + c;
                    sumAll += this->data[index];
                }
            }
            A.data[c] = T(sumAll);
        }
        return A;
    }
    else {
        // all
        Mat_<T> A(1, 1, 1);
        float sumAll = 0.;
        int length = rows * cols * channels;
        for (int i = 0; i < rows * cols * channels; i++) {
            sumAll += this->data[i];
        }
        A.data[0] = T(sumAll);
        return A;
    }

    return Mat_<T>();
}

/////////////////////////////////////////////////////
// 斜对角矩阵
template<typename T> inline
Mat_<T> diag(const Mat_<T>& A) {
    int lenght = BSJ_MAX(A.cols, A.rows);
    Mat_<T> B(lenght, lenght, 1);
    B.setZeros();
    for (int i = 0; i < lenght; i++) {
        B.data[i * lenght + i] = A.data[i];
    }
    return B;
}


// 乘法 一般是float
inline Mat2f operator*(Mat2f& A, float b) {
    int size = A.total();
    if (!size) {
        return A;
    }

    Mat2f B(A.rows, A.cols, A.channels);

    int step = A.cols * A.channels;

#ifdef USE_NEON
    float32x4_t _b = vdupq_n_f32(b);
#endif // USE_NEON

    for (int row = 0; row < A.rows; row++) {
        float* a_ptr = (float*)A.data + (row * A.cols) * A.channels;
        float* b_ptr = (float*)B.data + (row * A.cols) * A.channels;
#ifdef USE_NEON
        // float32 一次操作4个，不考虑越界
        int nn = step >> 2;
        int remain = step - (nn << 2);
        for (; nn > 0; nn--) {
            float32x4_t _pixel = vld1q_f32(a_ptr);
            float32x4_t _sum = vmulq_f32(_pixel, _b);
            vst1q_f32(b_ptr, _sum);
            a_ptr += 4;
            b_ptr += 4;
        }
#else
        int remain = step;
#endif // USE_NEON
        for (; remain > 0; remain--) {
            *b_ptr++ = *a_ptr++ * b;
        }
    }
    return B;
}

inline Mat operator*(Mat& A, float b) {
    int size = A.total();
    if (!size) {
        return A;
    }

    Mat B(A.rows, A.cols, A.channels);

    int step = A.cols * A.channels;
    unsigned char *a_ptr = (unsigned char *)A.data;
    unsigned char *b_ptr = (unsigned char *)B.data;
    for (int row = 0; row < A.rows; row++) {
        int remain = step;

        for (; remain > 0; remain--) {
            unsigned char value = BSJ_BETWEEN(*a_ptr * b, 0, 255);
            *b_ptr = value;
            a_ptr++;
            b_ptr++;
        }
    }
    return B;
}

inline Mat operator*(const Mat& A, float b) {
    int size = A.total();
    if (!size) {
        return A;
    }

    Mat B(A.rows, A.cols, A.channels);

    int step = A.cols * A.channels;
    unsigned char *a_ptr = (unsigned char *)A.data;
    unsigned char *b_ptr = (unsigned char *)B.data;
    for (int row = 0; row < A.rows; row++) {
        int remain = step;

        for (; remain > 0; remain--) {
            unsigned char value = BSJ_BETWEEN(*a_ptr * b, 0, 255);
            *b_ptr = value;
            a_ptr++;
            b_ptr++;
        }
    }
    return B;
}


template<typename T>
inline Mat_<T> operator/(const Mat_<T> &A, T b) {
    Mat_<T> tmp(A);
    for (int i = 0; i < tmp.rows * tmp.cols * tmp.channels; i++) {
        tmp.data[i] /= b;
    }
    return tmp;
}

template<typename T>
inline Mat_<T> operator+(const Mat_<T> &A, const Mat_<T> &B) {
    if (B.data == 0 || A.data == 0 || B.channels * B.rows != A.channels * A.rows || A.rows * A.cols * A.channels == 0 || B.rows * B.cols * B.channels == 0) {
        return Mat_<T>(0, 0, 0);
    }

    int cols = BSJ_MAX(B.cols, A.cols);
    int rows = A.rows;
    int channels = A.channels;
    Mat_<T> AB(rows, cols, channels);

    if (B.cols == A.cols) {
        for (int i = 0; i < AB.rows * AB.cols * AB.channels; i++) {
            AB.data[i] = A.data[i] + B.data[i];
        }
    } else {
        T *a;
        T *b;
        if (B.cols > A.cols) {
            a = (T *)B.data;
            b = (T *)A.data;
        } else {
            a = (T *)A.data;
            b = (T *)B.data;
        }

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int index = (i * cols + j) * channels + c;
                    AB.data[index] = a[index] + b[i + c];
                }
            }
        }
    }

    return AB;
}

template<typename T>
inline Mat_<T> operator-(const Mat_<T> &A, const Mat_<T> &B) {
    if (B.data == 0 || A.data == 0 || B.channels * B.cols != A.channels * A.cols || A.rows * A.cols * A.channels == 0 || B.rows * B.cols * B.channels == 0) {
        return Mat_<T>(0, 0, 0);
    }

    int cols = BSJ_MAX(B.cols, A.cols);
    int rows = A.rows;
    int channels = A.channels;
    Mat_<T> AB(rows, cols, channels);

    if (B.rows == A.rows) {
        for (int i = 0; i < AB.rows * AB.cols * AB.channels; i++) {
            AB.data[i] = A.data[i] - B.data[i];
        }
    } else {
        T *a;
        T *b;
        int coeff = 1;
        if (A.rows > B.rows) {
            a = (T *)A.data;
            b = (T *)B.data;
        } else {
            a = (T *)B.data;
            b = (T *)A.data;
            coeff = -1;
        }

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int index = (i * cols + j) * channels + c;
                    AB.data[index] = coeff * (a[index] - b[j + c]);
                }
            }
        }
    }

    return AB;
}

template<typename T>
inline Mat_<T> absdiff(const Mat_<T> &A, const Mat_<T> &B) {
    Mat_<T> AB(A);
    for (int i = 0; i < AB.rows * AB.cols * AB.channels; i++) {
        AB.data[i] = BSJ_ABS(A.data[i] - B.data[i]);
    }
    return AB;
}



//////////////////////////////////////////////////////////////

}
} // namespace BSJ_AI::CV
