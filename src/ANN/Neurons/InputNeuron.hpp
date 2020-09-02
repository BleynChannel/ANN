#pragma once

#include "../Neuron.hpp"

namespace net
{
    class InputNeuron : public Neuron
    {
        friend Layer;
    private:

    protected:
        virtual void handleInputData() {};
    public:
        InputNeuron() {};
    };
}