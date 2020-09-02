#include "OutputNeuron.hpp"

double net::OutputNeuron::getError() 
{
    return 0.0;
}

double net::OutputNeuron::getError(double correctOutput)
{
    return result - correctOutput;
}

net::OutputNeuron::OutputNeuron(double learningRate, const FunctionActiv& activFunct)
    : Neuron(learningRate, activFunct) {}