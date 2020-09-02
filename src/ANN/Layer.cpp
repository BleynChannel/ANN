#include "Layer.hpp"
#include "Net.hpp"

void net::Layer::addNet(Net* net)
{
    this->net = net;
    
    for (auto* neuron : neurons)
        neuron->net = net;
}

void net::Layer::setInitialized(bool initialized)
{
    if (net)
        net->initialized = initialized;
}

void net::Layer::addPrevLayer(Layer* layer)
{
    prevLayers.push_back(layer);
}

void net::Layer::addNextLayer(Layer* layer)
{
    nextLayers.push_back(layer);

    connect(layer->neurons);
}

void net::Layer::deletePrevLayer(Layer* layer)
{
    for (auto i = prevLayers.begin(); i != prevLayers.end(); i++)
        if (*i == layer)
            prevLayers.erase(i);
}

void net::Layer::deleteNextLayer(Layer* layer)
{
    for (auto i = nextLayers.begin(); i != nextLayers.end(); i++)
        if (*i == layer) {
            nextLayers.erase(i);
            deleteConnects(layer->neurons);
        }
}

void net::Layer::updateSpecialNeurons()
{
    input = false;
    output = false;

    inputNeurons.clear();
    outputNeurons.clear();

    InputNeuron* inputNeuron;
    OutputNeuron* outputNeuron;

    for (auto* neuron : neurons) {
        inputNeuron = dynamic_cast<InputNeuron*>(neuron);
        outputNeuron = dynamic_cast<OutputNeuron*>(neuron);

        if (inputNeuron != nullptr) {
            inputNeurons.push_back(inputNeuron);
            input = true;
        }
        if (outputNeuron != nullptr) {
            outputNeurons.push_back(outputNeuron);
            output = true;
        }
    }
}

bool net::Layer::updatePrevInput(bool main)
{
    prevInput = false;
    
    for (auto* layer : prevLayers)
        if (layer->updatePrevInput()) {
            prevInput = true;
            break;
        }

    if (!main && input)
        prevInput = true;

    return prevInput;
}

bool net::Layer::updatePrevOutput(bool main)
{
    prevOutput = false;
    
    for (auto* layer : nextLayers)
        if (layer->updatePrevOutput()) {
            prevOutput = true;
            break;
        }

    if (!main && output)
        prevOutput = true;

    return prevOutput;
}

void net::Layer::initializing()
{
    for (auto* neuron : neurons)
        neuron->init();

    updateSpecialNeurons();

    init();
}

void net::Layer::clearNeurons()
{
    for (auto* neuron : neurons)
        neuron->clear();

    clear();
}

void net::Layer::setDataInputNeurons(const std::vector<double>& inputs)
{
    uint32_t sizeInputs = inputs.size();
    InputNeuron* inputNeuron;

    if (sizeInputs != inputNeurons.size())
        Debug::error("The number of inputs is not equal to the number of input neurons");

    for (uint32_t i = 0; i < sizeInputs; i++) {
        inputNeuron = inputNeurons[i];

        inputNeuron->result = inputs[i];
        inputNeuron->handleInputData();
    }
}

std::vector<double> net::Layer::getDataOutputNeurons()
{
    std::vector<double> outputs;
    for (auto* neuron : outputNeurons) {
        neuron->handleOutputData();
        outputs.push_back(neuron->result);
    }
    return outputs;
}

void net::Layer::run(bool isMainLayer)
{
    if (isMainLayer) {
        for (auto* neuron : neurons)
            if (dynamic_cast<InputNeuron*>(neuron) == nullptr)
                neuron->query();
    } else
        for (auto* neuron : neurons)
            neuron->query();

    ready = true;

    if (!nextLayers.empty())
        for (auto* layer : nextLayers)
            layer->query();
}

void net::Layer::runTrain()
{
    ready = true;

    for (auto* layer : prevLayers)
        layer->train();
}

void net::Layer::addConnect(Neuron* prevNeuron, Neuron* nextNeuron)
{
    if (prevNeuron && nextNeuron) {
        Connect* connect = new Connect(prevNeuron, nextNeuron);

        prevNeuron->nextConnects.push_back(connect);
        nextNeuron->prevConnects.push_back(connect);
    } else
        Debug::warning("1 and/or 2 argument is nullptr");
}

void net::Layer::deleteConnect(Neuron* prevNeuron, Neuron* nextNeuron)
{
    if (prevNeuron && nextNeuron) {
        for (auto prevIter = prevNeuron->nextConnects.begin(); prevIter != prevNeuron->nextConnects.end(); prevIter++)
            for (auto nextIter = nextNeuron->prevConnects.begin(); nextIter != nextNeuron->prevConnects.end(); nextIter++)
                if (*prevIter == *nextIter) {
                    delete *prevIter;

                    prevNeuron->nextConnects.erase(prevIter);
                    nextNeuron->prevConnects.erase(nextIter);
                }
    } else
        Debug::warning("1 and/or 2 argument is nullptr");
}

void net::Layer::deleteConnects(const std::vector<Neuron*>& nextNeurons)
{
    for (auto* prevNeuron : neurons)
        for (auto* nextNeuron : nextNeurons)
            deleteConnect(prevNeuron, nextNeuron);
}

net::Layer::Layer() : net(nullptr), input(false), output(false), prevInput(false), prevOutput(false) {}

net::Layer::Layer(const std::vector<Neuron*>& neurons) 
: net(nullptr), input(false), output(false), prevInput(false), prevOutput(false)
{
    for (auto* neuron : neurons)
        addNeuron(neuron);
}

net::Layer::Layer(const std::vector<Neuron*>&& neurons) 
: net(nullptr), input(false), output(false), prevInput(false), prevOutput(false)
{
    for (auto* neuron : neurons)
        addNeuron(neuron);
}

net::Layer::~Layer()
{
    uint32_t size = neurons.size();
    for (uint32_t i = 0; i < size; i++)
        if (neurons[i]) {
            delete neurons[i];
            neurons[i] = nullptr;
        }
}

void net::Layer::deleteNeuron(Neuron* neuron)
{
    if (neuron) {
        for (auto i = neurons.begin(); i != neurons.end(); i++)
            if (*i == neuron) {
                if (net)
                    net->initialized = false;

                neurons.erase(i);
                delete neuron;
            }
    } else
        Debug::warning("Argument is nullptr");
}

void net::Layer::deleteNeuron(uint32_t index)
{
    if (index < neurons.size()) {
        auto iter = neurons.begin() + index;

        if (net)
            net->initialized = false;

        neurons.erase(iter);
        delete *iter;
    } else
        Debug::warning("Index is greater than numbers of neurons");
}

const net::Neuron* net::Layer::getNeuron(uint32_t index) const
{
    if (index < neurons.size()) {
        return neurons[index];
    } else
        Debug::warning("Index is greater than numbers of neurons");

    return nullptr;
}

const std::vector<net::Neuron*>& net::Layer::getNeurons() const
{
    return neurons;
}

const std::vector<net::Layer*>& net::Layer::getPrevLayers() const
{
    return prevLayers;
}

const std::vector<net::Layer*>& net::Layer::getNextLayers() const
{
    return nextLayers;
}

void net::Layer::query(bool isMainLayer)
{
    if (isMainLayer) {
        run(isMainLayer);
    } else {
        bool isNotReady = false;

        for (auto* layer : prevLayers)
            if (!layer->ready) {
                isNotReady = true;
                break;
            }

        if (!isNotReady)
            run();
    }
}

void net::Layer::train(const std::vector<double>& outputs)
{
    uint32_t sizeOutputs = outputs.size();

    if (sizeOutputs != outputNeurons.size())
        Debug::error("The number of outputs is not equal to the number of output neurons");

    OutputNeuron* neuron;

    for (uint32_t i = 0; i < sizeOutputs; i++) {
        neuron = outputNeurons[i];

        neuron->error = neuron->getError(outputs[i]);
        neuron->train(true);
    }

    runTrain();
}

void net::Layer::train(bool isMainLayer)
{
    if (isMainLayer) {
        uint32_t sizeOutputs = outputNeurons.size();
        OutputNeuron* neuron;

        for (uint32_t i = 0; i < sizeOutputs; i++) {
            neuron = outputNeurons[i];

            neuron->error = neuron->getError();
            neuron->train(true);
        }

        runTrain();
    } else {
        bool isNotReady = false;

        for (auto* layer : nextLayers)
            if (!layer->ready) {
                isNotReady = true;
                break;
            }

        if (!isNotReady) {
            for (auto* neuron : neurons)
                neuron->train();

            runTrain();
        }
    }
}