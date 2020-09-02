#pragma once

#include <iostream>
#include <exception>

namespace net
{
    class Debug
    {
    public:
        static bool debug;
    public:
        static void warning(const char* msg) noexcept;
        static void warning(const std::string& msg) noexcept;
        static void error(const char* msg);
        static void error(const std::string& msg);
    };
}