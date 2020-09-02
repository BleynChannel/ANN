#pragma once

#include <iostream>
#include <vector>

namespace net 
{
    class Neuron;

    struct Connect
    {
        friend Neuron;
    private:
        Neuron* prev;
        Neuron* next;
    
        double weight;
        double error;
    public:
        Connect(Neuron* prev, Neuron* next);

        const Neuron* getPrevNeuron() const;
        const Neuron* getNextNeuron() const;

        double getWeight() const;
        double getError() const;
    };
}