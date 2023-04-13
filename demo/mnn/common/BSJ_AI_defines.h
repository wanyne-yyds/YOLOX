#pragma once
#include <algorithm>
#include <vector>
#define BSJ_AI_FLAG_SUCCESSFUL      (0)  /*<! �ɹ� */
#define BSJ_AI_FLAG_BAD_PARAMETER   (-1) /*<! �������� */
#define BSJ_AI_FLAG_FAILED          (-2) /*<! ʧ��  */
#define BSJ_AI_FLAG_NULLPOINTER     (-3) /*<! ��ָ��  */
#define BSJ_AI_FLAG_UNINITIALIZED   (-4) /*<! δ���г�ʼ�� */
#define BSJ_AI_FLAG_INITIALIZATION  (-5) /*<! ���ڳ�ʼ�� */
#define BSJ_AI_FLAG_BUSY            (-6) /*<! ����ִ������ */
#define BSJ_AI_FLAG_LICENSE_INVALID (-7) /*<! ��Ч��Ȩ */
#define BSJ_AI_FLAG_EXCEPTION       (-8) /*<! �쳣 */

#define BSJ_AI_FLAG_NO_NECESSARY  (-11) /*<! ιͼ̫�죬���� */
#define BSJ_AI_FLAG_OUT_OF_ACTION (-12) /*<! ����ȡֵ��Χ֮�ڣ���ĳ���ٶ� */

#ifdef _WIN32
    #define BSJ_AI_API_PUBLIC
    #define BSJ_AI_API_LOCAL
#else
    #define BSJ_AI_API_PUBLIC __attribute((visibility("default")))
    #define BSJ_AI_API_LOCAL  __attribute((visibility("hidden")))
#endif

namespace BSJ_AI {

    /** ģʽ */
    typedef enum eRunningMode {
        M_NORMAL = 0, /*<! ��ͨģʽ */
        M_DEMO   = 1, /*<! ����ģʽ */
        M_JTT883 = 2  /*<! ����ģʽ */
    } RUNNING_MODE;

    /** ������ */
    typedef enum eSensitivity {
        OBTUSEST    = 1, /*<! �ǳ������� */
        OBTUSER     = 2, /*<! ������ */
        GENERAL     = 3, /*<! һ�� */
        SENSITIVER  = 4, /*<! ���� */
        SENSITIVEST = 5  /*<! �ǳ����� */
    } SENSITIVITY;

    /** ͼ���ʽ */
    typedef enum eImageFormat {
        YUYV   = 0,
        UYVY   = 1,
        NV21   = 2,
        I420   = 3,
        RGB565 = 4,
        BGR565 = 5,
        RGB888 = 6,
        BGR888 = 7,
        GRAY   = 8, /*<! �ݲ�֧�� */
        NV12   = 9,
        NV16   = 10, /*<! cpu�ݲ�֧�� */
    } IMAGE_FORMAT;

    /** ͼ�����ݽṹ */
    typedef struct stImageData {
        char        *data;       /*<! ����ָ�� */
        unsigned int dataLength; /*<! ͼ�񳤶� */
        unsigned int imgHeight;  /*<! ͼ��� */
        unsigned int imgWidth;   /*<! ͼ��� */
        IMAGE_FORMAT format;     /*<! ͼ���ʽ */

        /**
         * @brief   ���캯��.
         * @param   _len		ͼ�񳤶�
         * @param   _data       ����ָ��
         * @param   _h          ͼ���
         * @param   _w          ͼ���
         * @param   _f          ͼ���ʽ
         */
        stImageData(unsigned int _len = 0, char *_data = NULL, int _h = 0, int _w = 0, IMAGE_FORMAT _f = IMAGE_FORMAT::BGR888) {
            dataLength = _len;
            data       = _data;
            imgWidth   = _w;
            imgHeight  = _h;
            format     = _f;
        }

    } ImageData;

    /**
     * @brief    2D�㡣
     */
    template <typename _Tp>
    class Point_ {
    public:
        typedef _Tp value_type;

        Point_() :
            x(0), y(0){};
        Point_(_Tp _x, _Tp _y) :
            x(_x), y(_y){};
        Point_(const Point_ &pt) :
            x(pt.x), y(pt.y){};

        Point_ &operator=(const Point_ &pt) {
            x = pt.x;
            y = pt.y;
            return *this;
        };

        //! dot product
        _Tp dot(const Point_ &pt) const {
            return _Tp(x * pt.x + y * pt.y);
        };
        //! cross-product
        double cross(const Point_ &pt) const {
            return (double)x * pt.y - (double)y * pt.x;
        };

        _Tp x;
        _Tp y;
    };

    typedef Point_<int>    Point2i;
    typedef Point_<float>  Point2f;
    typedef Point_<double> Point2d;
    typedef Point2i        Point;

    /**
     * @brief    �ߴ硣
     */
    template <typename _Tp>
    class Size_ {
    public:
        typedef _Tp value_type;

        //! default constructor
        Size_() :
            width(0), height(0){};
        Size_(_Tp _width, _Tp _height) :
            width(_width), height(_height){};
        Size_(const Size_ &sz) :
            width(sz.width), height(sz.height){};
        Size_(const Point_<_Tp> &pt) :
            width(pt.x), height(pt.y){};

        Size_ &operator=(const Size_ &sz) {
            width  = sz.width;
            height = sz.height;
            return *this;
        };
        //! the area (width*height)
        _Tp area() const {
            const _Tp result = width * height;
            return result;
        };
        //! true if empty
        bool empty() const {
            return width <= 0 || height <= 0;
        };

        //! conversion of another data type.
        friend Size_<_Tp> &operator*=(Size_<_Tp> &a, _Tp b) {
            a.width *= b;
            a.height *= b;
            return a;
        }

        friend Size_<_Tp> operator*(const Size_<_Tp> &a, _Tp b) {
            Size_<_Tp> tmp(a);
            tmp *= b;
            return tmp;
        }

        friend Size_<_Tp> &operator/=(Size_<_Tp> &a, _Tp b) {
            a.width /= b;
            a.height /= b;
            return a;
        }

        friend Size_<_Tp> operator/(const Size_<_Tp> &a, _Tp b) {
            Size_<_Tp> tmp(a);
            tmp /= b;
            return tmp;
        }

        friend Size_<_Tp> &operator+=(Size_<_Tp> &a, const Size_<_Tp> &b) {
            a.width += b.width;
            a.height += b.height;
            return a;
        }

        friend Size_<_Tp> operator+(const Size_<_Tp> &a, const Size_<_Tp> &b) {
            Size_<_Tp> tmp(a);
            tmp += b;
            return tmp;
        }

        friend Size_<_Tp> &operator-=(Size_<_Tp> &a, const Size_<_Tp> &b) {
            a.width -= b.width;
            a.height -= b.height;
            return a;
        }

        friend Size_<_Tp> operator-(const Size_<_Tp> &a, const Size_<_Tp> &b) {
            Size_<_Tp> tmp(a);
            tmp -= b;
            return tmp;
        }

        friend bool operator==(const Size_<_Tp> &a, const Size_<_Tp> &b) {
            return a.width == b.width && a.height == b.height;
        }

        friend bool operator!=(const Size_<_Tp> &a, const Size_<_Tp> &b) {
            return !(a == b);
        }

        _Tp width;  //!< the width
        _Tp height; //!< the height
    };

    typedef Size_<int>    Size2i;
    typedef Size_<float>  Size2f;
    typedef Size_<double> Size2d;
    typedef Size2i        Size;

    /**
     * @brief    ����4�������࣬һ�����ڻ�ͼ��
     */
    template <typename _Tp>
    class Scalar_ {
    public:
        //! default constructor
        Scalar_() {
            val.resize(4, 0);
        };
        Scalar_(_Tp v0, _Tp v1, _Tp v2 = 0, _Tp v3 = 0) {
            val.resize(4, 0);
            val[0] = v0;
            val[1] = v1;
            val[2] = v2;
            val[3] = v3;
        };
        Scalar_(_Tp v0) {
            val.resize(4, 0);
            val[0] = v0;
        };

        std::vector<_Tp> val;
    };
    typedef Scalar_<int>    Scalar2i;
    typedef Scalar_<float>  Scalar2f;
    typedef Scalar_<double> Scalar2d;
    typedef Scalar2f        Scalar;

    /**
     * @brief    2D���ο�ģ�壬֧��x,y,w,h�������������½ǵ㡣
     */
    template <typename _Tp>
    class Rect_ {
    public:
        typedef _Tp value_type;

        Rect_() :
            x(0), y(0), width(0), height(0){};
        Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height) :
            x(_x), y(_y), width(_width), height(_height){};
        Rect_(const Point_<_Tp> &pt1, const Point_<_Tp> &pt2) {
            x      = std::min(pt1.x, pt2.x);
            y      = std::min(pt1.y, pt2.y);
            width  = std::max(pt1.x, pt2.x) - x;
            height = std::max(pt1.y, pt2.y) - y;
        };

        //! the top-left corner
        Point_<_Tp> tl() const {
            return Point_<_Tp>(x, y);
        };
        //! the bottom-right corner
        Point_<_Tp> br() const {
            return Point_<_Tp>(x + width, y + height);
        };

        //! size (width, height) of the rectangle
        Size_<_Tp> size() const {
            return Size_<_Tp>(width, height);
        }

        _Tp area() const {
            return this->width * this->height;
        };

        friend bool operator==(const Rect_<_Tp> &a, const Rect_<_Tp> &b) {
            return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
        }

        friend Rect_<_Tp> operator&=(Rect_<_Tp> &a, const Rect_<_Tp> &b) {
            //_Tp x1 = std::max(a.x, b.x);
            //_Tp y1 = std::max(a.y, b.y);
            // a.width = std::min(a.x + a.width, b.x + b.width) - x1;
            // a.height = std::min(a.y + a.height, b.y + b.height) - y1;
            // a.x = x1;
            // a.y = y1;
            // if (a.width <= 0 || a.height <= 0)
            //    a = Rect_();
            return a;
        }

        friend Rect_<_Tp> operator|=(Rect_<_Tp> &a, const Rect_<_Tp> &b) {
            // if (a.empty()) {
            //     a = b;
            // } else if (!b.empty()) {
            //     _Tp x1 = std::min(a.x, b.x);
            //     _Tp y1 = std::min(a.y, b.y);
            //     a.width = std::max(a.x + a.width, b.x + b.width) - x1;
            //     a.height = std::max(a.y + a.height, b.y + b.height) - y1;
            //     a.x = x1;
            //     a.y = y1;
            // }
            return a;
        }

        friend Rect_<_Tp> operator&(const Rect_<_Tp> &a, const Rect_<_Tp> &b) {
            Rect_<_Tp> c = a;
            return c &= b;
        }

        friend Rect_<_Tp> operator|(const Rect_<_Tp> &a, const Rect_<_Tp> &b) {
            Rect_<_Tp> c = a;
            return c |= b;
        }

        _Tp x;
        _Tp y;
        _Tp width;
        _Tp height;
    };
    typedef Rect_<int>    Rect2i;
    typedef Rect_<float>  Rect2f;
    typedef Rect_<double> Rect2d;
    typedef Rect2i        Rect;

} // namespace BSJ_AI