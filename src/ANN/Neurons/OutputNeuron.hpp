#pragma once

#include "../Neuron.hpp"

namespace net
{
    class OutputNeuron : public Neuron
    {
        friend Layer;
    private:

    protected:
        virtual double getError();
        virtual double getError(double correctOutput);

        virtual void handleOutputData() {};
    public:
        OutputNeuron(double learningRate = 0.0, const FunctionActiv& activFunct = FunctionActiv());
    };
}