#pragma once

#include "../Layer.hpp"

namespace net
{
    class EverWithEver : public Layer
    {
    protected:
        virtual void connect(const std::vector<Neuron*>& nextNeurons) override;
    public:
        EverWithEver() {}

        template <typename NeuronType>
        EverWithEver(Array<NeuronType>& neurons) : Layer(neurons) {}
        template <typename NeuronType>
        EverWithEver(Array<NeuronType>&& neurons) : Layer(neurons) {}

        EverWithEver(const std::vector<Neuron*>& neurons) : Layer(neurons) {}
        EverWithEver(const std::vector<Neuron*>&& neurons) : Layer(neurons) {}

        template <typename Function>
        EverWithEver(uint32_t count, Function function) : Layer(count, function) {}
    };
}