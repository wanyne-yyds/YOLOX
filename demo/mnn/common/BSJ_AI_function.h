#pragma once

#include <iostream>
#include <functional>

namespace BSJ_AI {
    class noncopyable_t {
    protected:
        noncopyable_t(void)                             = default;
        noncopyable_t(noncopyable_t &&)                 = default;
        noncopyable_t(const noncopyable_t &)            = delete;
        noncopyable_t &operator=(const noncopyable_t &) = delete;
        ~noncopyable_t(void)                            = default;
    };

    // ??????????????return?????
    template <class T>
    class defer : public noncopyable_t {
    public:
        defer(std::function<T> f) {
            func      = f;
            will_call = true;
        }

        ~defer(void) noexcept {
            if (will_call) {
                try {
                    (void)func();
                } catch (...) {}
            }
        }
        /* This helps prevent some "unused variable" warnings under, for instance,
         * GCC 3.2.
         */
        void touch(void) const {
        }

        /*
         * defer ???????? if ????????????
         * ???if?????????§Ö?????return, ??????????
         */

        void cancel_call(void) noexcept {
            will_call = false;
        }

    private:
        std::function<T> func;
        bool             will_call;
    };

} // namespace BSJ_AI