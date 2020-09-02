#include "EverWithEver.hpp"

void net::EverWithEver::connect(const std::vector<Neuron*>& nextNeurons)
{
    for (auto* prevNeuron : neurons)
        for (auto* nextNeuron : nextNeurons)
            addConnect(prevNeuron, nextNeuron);
}