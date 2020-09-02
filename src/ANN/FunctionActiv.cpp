#include "FunctionActiv.hpp"
#include <math.h>

net::FunctionActiv::FunctionActiv(const std::function<double(double)>& activ, const std::function<double(double)>& d_activ)
    : activ(activ), d_activ(d_activ) {}

double net::FunctionActiv::sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

double net::FunctionActiv::d_sigmoid(double x)
{
    return x * (1.0 - x);
}

double net::FunctionActiv::relu(double x)
{
    return std::fmax(x, 0.0);
}

double net::FunctionActiv::d_relu(double x)
{
    return static_cast<double>(x > 0);
}

double net::FunctionActiv::tanh(double x)
{
    return std::tanh(x);
}

double net::FunctionActiv::d_tanh(double x)
{
    return 1.0 - x * x;
}