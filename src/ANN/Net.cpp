#include "Net.hpp"

double net::Net::minRandom = -1.0;
double net::Net::maxRandom = 1.0;

void net::Net::updateSpecialLayers()
{
    inputLayers.clear();
    outputLayers.clear();

    for (auto* layer : layers) {
        if (layer->input)
            inputLayers.push_back(layer);
        if (layer->output)
            outputLayers.push_back(layer);
    }
}

void net::Net::updateRunningLayers()
{
    for (auto* layer : inputLayers)
        layer->updatePrevInput(true);

    for (auto* layer : outputLayers)
        layer->updatePrevOutput(true);
}

void net::Net::clearLayers()
{
    for (auto* layer : layers) {
        layer->ready = false;
        layer->clearNeurons();
    }

    clear();
}

void net::Net::clearReady()
{
    for (auto* layer : layers)
        layer->ready = false;
}

void net::Net::algorithmInput(const std::vector<double>& inputs, Layer* mainInputLayer)
{
    if (!inputLayers.empty()) {
        if (!mainInputLayer)
            mainInputLayer = inputLayers[0];

        mainInputLayer->setDataInputNeurons(inputs);
        algorithm(mainInputLayer);
    }
}

void net::Net::algorithmInput(const std::vector<std::vector<double>>& inputs, const std::vector<Layer*>& mainInputLayers)
{
    uint32_t sizeInputs = inputs.size();

    const std::vector<double>* input;
    Layer* inputLayer;

    if (mainInputLayers.empty()) {
        if (sizeInputs != inputLayers.size())
            Debug::error("The number of inputs is not equal to the number of input layers");

        for (uint32_t i = 0; i < sizeInputs; i++) {
            input = &inputs[i];
            inputLayer = inputLayers[i];

            inputLayer->setDataInputNeurons(*input);
            algorithm(inputLayer);
        }
    } else {
        if (sizeInputs != mainInputLayers.size())
            Debug::error("The number of inputs is not equal to the number of listed input layers");

        for (uint32_t i = 0; i < sizeInputs; i++) {
            input = &inputs[i];
            inputLayer = mainInputLayers[i];

            if (!inputLayer) {
                inputLayer->setDataInputNeurons(*input);
                algorithm(inputLayer);
            }
        }
    }
}

std::vector<double> net::Net::algorithmOutput(Layer* mainOutputLayer)
{
    if (!outputLayers.empty()) {
        if (!mainOutputLayer)
            mainOutputLayer = outputLayers[0];
        
        return mainOutputLayer->getDataOutputNeurons();
    }

    return std::vector<double>();
}

std::vector<std::vector<double>> net::Net::algorithmOutput(const std::vector<Layer*>& mainOutputLayers)
{
    std::vector<std::vector<double>> outputs;

    if (mainOutputLayers.empty()) {
        for (auto* layer : outputLayers)
            outputs.push_back(layer->getDataOutputNeurons());
    } else {
        for (auto* layer : mainOutputLayers)
            if (layer)
                outputs.push_back(layer->getDataOutputNeurons());
    }

    return outputs;
}

void net::Net::algorithm(Layer* layer)
{
    if (!layer->prevInput)
        layer->query(true);
}

void net::Net::algorithmTrain(Layer* mainOutputLayer, const std::vector<double>* correctOutputs)
{
    if (!outputLayers.empty()) {
        if (!mainOutputLayer)
            mainOutputLayer = outputLayers[0];

        if (correctOutputs) {
            if (!mainOutputLayer->prevOutput)
                mainOutputLayer->train(*correctOutputs);
        } else
            if (!mainOutputLayer->prevOutput)
                mainOutputLayer->train(true);
    }
}

void net::Net::algorithmTrain(const std::vector<Layer*>& mainOutputLayers, const std::vector<std::vector<double>>* correctOutputs)
{
    uint32_t sizeOutputs;
    Layer* outputLayer;

    if (mainOutputLayers.empty()) {
        if (correctOutputs) {
            sizeOutputs = correctOutputs->size();

            if (sizeOutputs != outputLayers.size())
                Debug::error("The number of outputs is not equal to the number of output layers");

            for (uint32_t i = 0; i < sizeOutputs; i++) {
                outputLayer = outputLayers[i];

                if (!outputLayer->prevOutput)
                    outputLayer->train((*correctOutputs)[i]);
            }
        } else {
            sizeOutputs = outputLayers.size();

            for (uint32_t i = 0; i < sizeOutputs; i++) {
                outputLayer = outputLayers[i];

                if (!outputLayer->prevOutput)
                    outputLayer->train(true);
            }
        }
    } else {
        if (correctOutputs) {
            sizeOutputs = correctOutputs->size();

            if (sizeOutputs != mainOutputLayers.size())
                Debug::error("The number of outputs is not equal to the number of listed output layers");

            for (uint32_t i = 0; i < sizeOutputs; i++) {
                outputLayer = mainOutputLayers[i];

                if (!outputLayer->prevOutput)
                    outputLayer->train((*correctOutputs)[i]);
            }
        } else {
            sizeOutputs = mainOutputLayers.size();

            for (uint32_t i = 0; i < sizeOutputs; i++) {
                outputLayer = mainOutputLayers[i];

                if (!outputLayer->prevOutput)
                    outputLayer->train(true);
            }
        }
    }
}

net::Net::Net() : initialized(false) {}

net::Net::~Net()
{    
    uint32_t size = layers.size();
    for (uint32_t i = 0; i < size; i++)
        if (layers[i]) {
            delete layers[i];
            layers[i] = nullptr;
        }
}

void net::Net::deleteLayer(Layer* layer)
{
    if (layer) {
        for (auto i = layers.begin(); i != layers.end(); i++)
            if (*i == layer) {
                for (auto* prevLayer : layer->prevLayers)
                    prevLayer->deleteNextLayer(layer);

                layers.erase(i);
                delete layer;

                initialized = false;
            }
    } else
        Debug::warning("1 argument is nullptr");
}

void net::Net::deleteLayer(uint32_t index)
{
    if (index < layers.size()) {
        auto iter = (layers.begin() + index);
        Layer* layer = *iter;

        for (auto* prevLayer : layer->prevLayers)
            prevLayer->deleteNextLayer(layer);

        layers.erase(iter);
        delete layer;

        initialized = false;
    } else
        Debug::warning("Index is greater than numbers of layers");
}

const net::Layer* net::Net::getLayer(uint32_t index) const
{
    if (index < layers.size()) {
        return layers[index];
    } else
        Debug::warning("Index is greater than numbers of layers");

    return nullptr;
}

const std::vector<net::Layer*>& net::Net::getLayers() const
{
    return layers;
}

void net::Net::initializing()
{
    for (auto* layer : layers)
        layer->initializing();

    updateSpecialLayers();
    updateRunningLayers();

    init();

    initialized = true;
}

std::vector<double> net::Net::query(
    const std::vector<double>& inputs, Layer* mainInputLayer, Layer* mainOutputLayer
)
{
    if (!initialized)
        initializing();
    clearLayers();
    
    algorithmInput(inputs, mainInputLayer);
    return algorithmOutput(mainOutputLayer);
}

std::vector<double> net::Net::query(
    const std::vector<std::vector<double>>& inputs, const std::vector<Layer*>& mainInputLayers, Layer* mainOutputLayer
)
{
    if (!initialized)
        initializing();
    clearLayers();

    algorithmInput(inputs, mainInputLayers);
    return algorithmOutput(mainOutputLayer);
}

std::vector<std::vector<double>> net::Net::query(
    const std::vector<double>& inputs, Layer* mainInputLayer, const std::vector<Layer*>& mainOutputLayers
)
{
    if (!initialized)
        initializing();
    clearLayers();

    algorithmInput(inputs, mainInputLayer);
    return algorithmOutput(mainOutputLayers);
}

std::vector<std::vector<double>> net::Net::query(
    const std::vector<std::vector<double>>& inputs, const std::vector<Layer*>& mainInputLayers, 
    const std::vector<Layer*>& mainOutputLayers
)
{
    if (!initialized)
        initializing();
    clearLayers();

    algorithmInput(inputs, mainInputLayers);
    return algorithmOutput(mainOutputLayers);
}

void net::Net::train(const std::vector<double>& correctOutputs, Layer* mainOutputLayer)
{
    if (!initialized)
        initializing();
    clearReady();

    algorithmTrain(mainOutputLayer, &correctOutputs);
}

void net::Net::train(const std::vector<std::vector<double>>& correctOutputs, const std::vector<Layer*>& mainOutputLayers)
{
    if (!initialized)
        initializing();
    clearReady();

    algorithmTrain(mainOutputLayers, &correctOutputs);
}

void net::Net::train(Layer* mainOutputLayer)
{
    if (!initialized)
        initializing();
    clearReady();

    algorithmTrain(mainOutputLayer);
}

void net::Net::train(const std::vector<Layer*>& mainOutputLayers)
{
    if (!initialized)
        initializing();
    clearReady();

    algorithmTrain(mainOutputLayers);
}