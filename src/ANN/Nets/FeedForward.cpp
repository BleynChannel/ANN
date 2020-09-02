#include "FeedForward.hpp"
#include "../Layers/EverWithEver.hpp"
#include "../Neurons/BiasNeuron.hpp"

net::TopologyLayer::TopologyLayer(uint32_t countNeuron, double learningRate, const FunctionActiv& function, bool bias)
    : countNeuron(countNeuron), learningRate(learningRate), function(function), bias(bias) {}

net::FeedForward::FeedForward(const std::vector<TopologyLayer>& topology)
{
    uint32_t sizeTopology = topology.size();
    Layer* prevLayer;

    for (uint32_t i = 0; i < sizeTopology; i++) {
        const TopologyLayer& tmp = topology[i];

        if (!i) {
            Array<InputNeuron> array(tmp.countNeuron);

            if (tmp.bias)
                array.neurons.push_back(new BiasNeuron);

            prevLayer = addLayer(new EverWithEver(array));
        } else if (i + 1 == sizeTopology) {
            Array<OutputNeuron> array(tmp.countNeuron, tmp.learningRate, tmp.function);

            if (tmp.bias)
                array.neurons.push_back(new BiasNeuron);

            addLayer(new EverWithEver(array), prevLayer);
        } else {
            Array<Neuron> array(tmp.countNeuron, tmp.learningRate, tmp.function);

            if (tmp.bias)
                array.neurons.push_back(new BiasNeuron);

            prevLayer = addLayer(new EverWithEver(array), prevLayer);
        }
    }
}