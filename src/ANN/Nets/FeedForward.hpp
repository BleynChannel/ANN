#pragma once

#include "../Net.hpp"

namespace net 
{
    struct TopologyLayer
    {
    public:
        uint32_t countNeuron;
        double learningRate;
        FunctionActiv function;
        bool bias;
    public:
        TopologyLayer(uint32_t countNeuron, double learningRate, const FunctionActiv& function = FunctionActiv(), bool bias = false);
    };

    class FeedForward : public Net
    {
    public:
        FeedForward(const std::vector<TopologyLayer>& topology);
    };
}