#include "Debug.hpp"

bool net::Debug::debug = true;

void net::Debug::warning(const char* msg) noexcept
{
    if (debug)
        std::cout << "WARNING: " << msg << std::endl;
}

void net::Debug::warning(const std::string& msg) noexcept
{
    if (debug)
        std::cout << "WARNING: " << msg << std::endl;
}

void net::Debug::error(const char* msg)
{
    if (debug)
        std::cout << "ERROR: " << msg << std::endl;

    throw std::runtime_error(std::string("ERROR: ") + msg);
}

void net::Debug::error(const std::string& msg)
{
    if (debug)
        std::cout << "ERROR: " << msg << std::endl;

    throw std::runtime_error("ERROR: " + msg);
}