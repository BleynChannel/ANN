#include "Neuron.hpp"

void net::Neuron::algorithmNeuron()
{
    result = 0.0;

    uint32_t size = prevConnects.size();
    Connect* prevConnect;

    for (uint32_t i = 0; i < size; i++) {
        prevConnect = prevConnects[i];

        result += prevConnect->weight * prevConnect->prev->result;
    }

    if (activFunct.activ)
        result = activFunct.activ(result);
}

double net::Neuron::algorithmTrain(Neuron* prev)
{
    double delta = learningRate * error * prev->result;
    
    if (activFunct.d_activ)
        delta *= activFunct.d_activ(result);

    return delta;
}

net::Neuron::Neuron(double learningRate, const FunctionActiv& activFunct) 
    : net(nullptr), layer(nullptr), learningRate(learningRate), activFunct(activFunct), error(0.0) {}

net::Neuron::~Neuron()
{
    for (auto* connect : nextConnects)
        delete connect;
}

const std::vector<net::Connect*> net::Neuron::getPrevConnects() const
{
    return prevConnects;
}

const std::vector<net::Connect*> net::Neuron::getNextConnects() const
{
    return nextConnects;
}

const net::FunctionActiv& net::Neuron::getActivationFunction() const
{
    return activFunct;
}

double net::Neuron::getLearningRate() const
{
    return learningRate;
}

double net::Neuron::getResult() const
{
    return result;
}

double net::Neuron::getError() const
{
    return error;
}

void net::Neuron::setActivationFunction(const FunctionActiv& activationFunction)
{
    activFunct = activationFunction;
}

void net::Neuron::setLearningRate(double learningRate)
{
    this->learningRate = learningRate;
}

void net::Neuron::query()
{
    algorithmNeuron();
}

void net::Neuron::train(bool isMain)
{
    if (!isMain) {
        error = 0.0;

        for (auto* connect : nextConnects)
            error += connect->error;
    }

    for (auto* connect : prevConnects)
        connect->error = error * connect->weight;

    for (auto* connect : prevConnects)
        connect->weight -= algorithmTrain(connect->prev);
}