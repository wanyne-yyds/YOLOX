#include "Mat.h"
#include "Mat_drawing_font.h"

namespace BSJ_AI {
namespace CV {
int circle(Mat &img, Point center, int radius, const Scalar &color, int thickness) {
    if (img.empty() || center.x <= 0 || center.y <= 0) {
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (thickness == -1) {
        for (int y = center.y - (radius - 1); y < center.y + radius; y++) {
            if (y < 0) continue;
            if (y >= img.rows) break;

            for (int x = center.x - (radius - 1); x < center.x + radius; x++) {
                if (x < 0) continue;
                if (x >= img.cols) break;

                int index = (y * img.cols + x) * img.channels;
                for (int c = 0; c < img.channels; c++) {
                    img.data[index + c] = color.val[c];
                }
            }
        }

        return BSJ_AI_FLAG_SUCCESSFUL;
    }

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    for (int y = center.y - radius - t0; y < center.y + radius + t1; y++) {
        if (y < 0) continue;
        if (y >= img.rows) break;

        for (int x = center.x - radius - t0; x < center.x + radius + t1; x++) {
            if (x < 0) continue;
            if (x >= img.rows) break;

            // distance from cx cy
            int index = (y * img.cols + x) * img.channels;
            for (int c = 0; c < img.channels; c++) {
                img.data[index + c] = color.val[c];
            }
        }
    }

    return BSJ_AI_FLAG_SUCCESSFUL;
}

int rectangle(Mat &img, BSJ_AI::Rect rect, const Scalar &color, int thickness) {
    int ret = BSJ_AI_FLAG_SUCCESSFUL;
    ret = rectangle(img, rect.tl(), rect.br(), color, thickness);
    return ret;
}

int rectangle(Mat &img, BSJ_AI::Point pt1, BSJ_AI::Point pt2, const Scalar &color, int thickness) {
    if (img.empty()) {
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (pt1.x >= pt2.x || pt1.y > pt2.y) {
        LOGE("BSJ_AI::CV::rectangle pt1.x[%d] >= pt2.x[%d] || pt1.y[%d] > pt2.y[%d]\n", pt1.x, pt2.x, pt1.y, pt2.y);
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }

    if (thickness == -1) {
        for (int y = pt1.y; y < pt2.y; y++) {
            if (y < 0) continue;
            if (y >= img.rows) break;

            for (int x = pt1.x; x < pt2.x; x++) {
                if (x < 0) continue;
                if (x >= img.cols) break;

                int index = (y * img.cols + x) * img.channels;
                for (int c = 0; c < img.channels; c++) {
                    img.data[index + c] = color.val[c];
                }
            }
        }

        return BSJ_AI_FLAG_SUCCESSFUL;
    }

    const int t0 = thickness / 2;
    const int t1 = thickness - t0;
    int rw = pt2.x - pt1.x;
    int rh = pt2.y - pt1.y;
    // draw top
    {
        for (int y = pt1.y - t0; y < pt1.y + t1; y++) {
            if (y < 0) continue;
            if (y >= img.rows) break;

            for (int x = pt1.x - t0; x < pt1.x + rw + t1; x++) {
                if (x < 0) continue;
                if (x >= img.cols) break;

                int index = (y * img.cols + x) * img.channels;
                for (int c = 0; c < img.channels; c++) {
                    img.data[index + c] = color.val[c];
                }
            }
        }
    }

    // draw bottom
    {
        for (int y = pt1.y + rh - t0; y < pt1.y + rh + t1; y++) {
            if (y < 0) continue;
            if (y >= img.rows) break;

            for (int x = pt1.x - t0; x < pt1.x + rw + t1; x++) {
                if (x < 0) continue;
                if (x >= img.cols) break;

                int index = (y * img.cols + x) * img.channels;
                for (int c = 0; c < img.channels; c++) {
                    img.data[index + c] = color.val[c];
                }
            }
        }
    }

    // draw left
    for (int x = pt1.x - t0; x < pt1.x + t1; x++) {
        if (x < 0) continue;
        if (x >= img.cols) break;

        for (int y = pt1.y + t1; y < pt1.y + rh - t0; y++) {
            if (y < 0) continue;
            if (y >= img.rows) break;

            int index = (y * img.cols + x) * img.channels;
            for (int c = 0; c < img.channels; c++) {
                img.data[index + c] = color.val[c];
            }
        }
    }

    // draw right
    for (int x = pt1.x + rw - t0; x < pt1.x + rw + t1; x++) {
        if (x < 0) continue;
        if (x >= img.cols) break;

        for (int y = pt1.y + t1; y < pt1.y + rh - t0; y++) {
            if (y < 0) continue;
            if (y >= img.rows) break;

            int index = (y * img.cols + x) * img.channels;
            for (int c = 0; c < img.channels; c++) {
                img.data[index + c] = color.val[c];
            }
        }
    }
    return BSJ_AI_FLAG_SUCCESSFUL;
}

Size getTextSize(const std::string text, int thickness) {
    // »ñÈ¡×ÖÌå´óÐ¡
    int w = 0;
    int h = 0;

    const int n = text.size();

    int line_w = 0;
    for (int i = 0; i < n; i++) {
        char ch = text[i];

        if (ch == '\n') {
            // newline
            w = BSJ_MAX(w, line_w);
            h += thickness * 2;
            line_w = 0;
        }

        if (isprint(ch) != 0) {
            line_w += thickness;
        }
    }

    w = BSJ_MAX(w, line_w);
    h += thickness * 2;

    return Size(w, h);
}

int draw_text(Mat &img, const std::string text, BSJ_AI::Point pt, const Scalar &color, int thickness) {
    if (img.empty()) {
        return BSJ_AI_FLAG_BAD_PARAMETER;
    }
    const int n = text.size();
    int cursor_x = pt.x;
    int cursor_y = pt.y;
    for (int i = 0; i < n; i++) {
        char ch = text[i];

        if (ch == '\n') {
            // newline
            cursor_x = pt.x;
            cursor_y += thickness * 2;
        }
        if (isprint(ch) != 0) {
            int font_bitmap_index = ch - ' ';
            const unsigned char* font_bitmap = mono_font_data[font_bitmap_index];
            Mat mat_font(40, 20, 1, (unsigned char*)font_bitmap);
            Mat resize_font;
            resize(mat_font, resize_font, Size(thickness, thickness * 2));

            for (int j = cursor_y; j < cursor_y + thickness * 2; j++) {
                if (j < 0) continue;
                if (j >= img.rows) break;

                const unsigned char* palpha = resize_font.data + (j - cursor_y) * thickness;
                unsigned char* p = img.data + img.cols * img.channels * j;
                for (int k = cursor_x; k < cursor_x + thickness; k++) {
                    unsigned char alpha = palpha[k - cursor_x];

                    for (int c = 0; c < img.channels; c++) {
                        p[k * img.channels + c] = (int(p[k * img.channels + c]) * (255 - alpha) + color.val[c] * int(alpha)) / 255;
                    }
                }
            }

            cursor_x += thickness;
        }
    }
    return BSJ_AI_FLAG_SUCCESSFUL;
}

}
} // namespace BSJ_AI::CV