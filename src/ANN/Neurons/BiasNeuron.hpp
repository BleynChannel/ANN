#pragma once

#include "../Neuron.hpp"

namespace net
{
    class BiasNeuron : public Neuron
    {
    protected:
        virtual void algorithmNeuron();
        virtual double algorithmTrain(Neuron*);
    public:
        BiasNeuron();
    };
}