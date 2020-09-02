#pragma once

#include "Neurons/InputNeuron.hpp"
#include "Neurons/OutputNeuron.hpp"

namespace net
{
    class Layer;
    class Net;

    template <typename NeuronType>
    struct Array 
    {
        friend Layer;
    private:
        bool isDelete;
    public:
        std::vector<Neuron*> neurons;
    public:
        template <typename ...Args>
        Array(uint32_t count, Args... args) : isDelete(true)
        {
            for (uint32_t i = 0; i < count; i++)
                neurons.push_back(new NeuronType(args...));
        }
        ~Array()
        {
            if (isDelete)
                for (auto* neuron : neurons)
                    delete neuron;
        }
    };

    class Layer
    {
        friend Net;
    private:
        std::vector<InputNeuron*> inputNeurons;
        std::vector<OutputNeuron*> outputNeurons;
    private:
        bool input;
        bool output;

        bool prevInput;
        bool prevOutput;

        bool ready;
    protected:
        Net* net;

        std::vector<Layer*> prevLayers;
        std::vector<Layer*> nextLayers;

        std::vector<Neuron*> neurons;
    private:
        void addNet(Net* net);

        void setInitialized(bool initialized);

        void addPrevLayer(Layer* layer);
        void addNextLayer(Layer* layer);

        void deletePrevLayer(Layer* layer);
        void deleteNextLayer(Layer* layer);

        void updateSpecialNeurons();
        bool updatePrevInput(bool main = false);
        bool updatePrevOutput(bool main = false);
    private:
        void initializing();

        void clearNeurons();

        void setDataInputNeurons(const std::vector<double>& inputs);
        std::vector<double> getDataOutputNeurons();

        void run(bool isMainLayer = false);
        void runTrain();
    protected:
        void addConnect(Neuron* prevNeuron, Neuron* nextNeuron);
        void deleteConnect(Neuron* prevNeuron, Neuron* nextNeuron);
        void deleteConnects(const std::vector<Neuron*>& nextNeurons);

        virtual void connect(const std::vector<Neuron*>& nextNeurons) = 0;
        virtual void init() {};
        virtual void clear() {};
    public:
        Layer();

        template <typename NeuronType>
        Layer(Array<NeuronType>& neurons);
        template <typename NeuronType>
        Layer(Array<NeuronType>&& neurons);

        Layer(const std::vector<Neuron*>& neurons);
        Layer(const std::vector<Neuron*>&& neurons);

        template <typename Function>
        Layer(uint32_t count, Function function);

        virtual ~Layer();

        template <typename NeuronType>
        NeuronType* addNeuron(NeuronType* neuron);
        template <typename NeuronType>
        NeuronType* addNeuron(uint32_t index, NeuronType* neuron);

        void deleteNeuron(Neuron* neuron);
        void deleteNeuron(uint32_t index);

        const Neuron* getNeuron(uint32_t index) const;
        const std::vector<Neuron*>& getNeurons() const;

        virtual const std::vector<Layer*>& getPrevLayers() const;
        virtual const std::vector<Layer*>& getNextLayers() const;

        void query(bool isMainLayer = false);
        
        void train(const std::vector<double>& outputs);
        void train(bool isMainLayer = false);
    };

    template <typename NeuronType>
    Layer::Layer(Array<NeuronType>& neurons) 
    : net(nullptr), input(false), output(false), prevInput(false), prevOutput(false)
    {
        neurons.isDelete = false;

        for (auto* neuron : neurons.neurons)
            addNeuron(neuron);
    }

    template <typename NeuronType>
    Layer::Layer(Array<NeuronType>&& neurons) 
    : net(nullptr), input(false), output(false), prevInput(false), prevOutput(false)
    {
        neurons.isDelete = false;

        for (auto* neuron : neurons.neurons)
            addNeuron(neuron);
    }

    template <typename Function>
    Layer::Layer(uint32_t count, Function function) 
    : net(nullptr), input(false), output(false), prevInput(false), prevOutput(false)
    {
        for (uint32_t i = 0; i < count; i++)
            addNeuron(function(i));
    }

    template <typename NeuronType>
    NeuronType* Layer::addNeuron(NeuronType* neuron)
    {
        if (neuron) {
            neuron->net = net;
            neuron->layer = this;

            setInitialized(false);

            neurons.push_back(neuron);
        } else
            Debug::warning("Argument is nullptr");

        return neuron;
    }

    template <typename NeuronType>
    NeuronType* Layer::addNeuron(uint32_t index, NeuronType* neuron)
    {
        if (neuron)
            if (index < neurons.size()) {
                if (net)
                    neuron->net = net;
                neuron->layer = this;

                setInitialized(false);

                neurons.insert(neurons.begin() + index, neuron);
            } else
                Debug::warning("Index is greater than numbers of neurons");
        else
            Debug::warning("2 argument is nullptr");

        return neuron;
    }
}