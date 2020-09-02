#pragma once

#include <functional>

namespace net
{
    struct FunctionActiv
    {
    public:
        std::function<double(double)> activ;
        std::function<double(double)> d_activ;

        FunctionActiv(const std::function<double(double)>& activ = nullptr, const std::function<double(double)>& d_activ = nullptr);
    public:
        static double sigmoid(double x);
        static double d_sigmoid(double x);
        
        static double relu(double x);
        static double d_relu(double x);

        static double tanh(double x);
        static double d_tanh(double x);
    };
}