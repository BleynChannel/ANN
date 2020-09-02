#include "Connect.hpp"
#include "Net.hpp"

net::Connect::Connect(Neuron* prev, Neuron* next)
    : prev(prev), next(next), error(0.0)
{
    weight = rand() / static_cast<double>(RAND_MAX) * (Net::maxRandom - Net::minRandom) + Net::minRandom;
}

const net::Neuron* net::Connect::getPrevNeuron() const
{
    return prev;
}

const net::Neuron* net::Connect::getNextNeuron() const
{
    return next;
}

double net::Connect::getWeight() const
{
    return weight;
}

double net::Connect::getError() const
{
    return error;
}