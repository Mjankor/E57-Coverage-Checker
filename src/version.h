// What build is this?
//
// Two rounds of diagnosis were spent on output from a binary that predated the
// fixes being discussed, and there was no way to tell from the output itself.
// Every report the tool produces now says which build produced it.
//
// CMake supplies the git revision. Xcode does not, so it falls back to the
// translation unit's compile time — less precise, but enough to answer "is this
// the binary I built five minutes ago or the one from last week?", which is the
// question that actually gets asked.

#pragma once

#include <string>

namespace ver {

inline const char* revision() {
#ifdef E57COV_REV
    return E57COV_REV;
#else
    return "";
#endif
}

inline const char* buildStamp() { return __DATE__ " " __TIME__; }

// A one-line identification, e.g. "0269eb6 (built Sep  9 2026 11:20:14)".
inline const char* describe() {
    static const std::string s = [] {
        std::string r = revision();
        if (r.empty()) r = "no revision recorded";
        return r + " (built " + buildStamp() + ")";
    }();
    return s.c_str();
}

} // namespace ver
