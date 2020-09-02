#pragma once

#include "Layer.hpp"

namespace net
{
    class Net
    {
        friend Layer;
    private:
        std::vector<Layer*> inputLayers;
        std::vector<Layer*> outputLayers;

        bool initialized;
    protected:
        std::vector<Layer*> layers;
    public:
        static double minRandom;
        static double maxRandom;
    private:
        void updateSpecialLayers();
        void updateRunningLayers();
    private:
        void clearLayers();
        void clearReady();

        void algorithmInput(const std::vector<double>& inputs, Layer* mainInputLayer);
        void algorithmInput(const std::vector<std::vector<double>>& inputs, const std::vector<Layer*>& mainInputLayers);
        std::vector<double> algorithmOutput(Layer* mainOutputLayer);
        std::vector<std::vector<double>> algorithmOutput(const std::vector<Layer*>& mainOutputLayers);

        void algorithm(Layer* layer);
    private:
        void algorithmTrain(Layer* mainOutputLayer, const std::vector<double>* correctOutputs = nullptr);
        void algorithmTrain(const std::vector<Layer*>& mainOutputLayers, const std::vector<std::vector<double>>* correctOutputs = nullptr);
    protected:
        virtual void init() {};
        virtual void clear() {};
    public:
        Net();
        virtual ~Net();

        template <typename LayerType>
        LayerType* addLayer(LayerType* layer, Layer* prevLayer = nullptr);
        template <typename LayerType>
        LayerType* addLayer(LayerType* layer, std::vector<Layer*> prevLayers);

        void deleteLayer(Layer* layer);
        void deleteLayer(uint32_t index);

        const Layer* getLayer(uint32_t index) const;
        const std::vector<Layer*>& getLayers() const;

        void initializing();

        std::vector<double> query(
            const std::vector<double>& inputs, 
            Layer* mainInputLayer = nullptr, 
            Layer* mainOutputLayer = nullptr
        );
        std::vector<double> query(
            const std::vector<std::vector<double>>& inputs, 
            const std::vector<Layer*>& mainInputLayers = std::vector<Layer*>(), 
            Layer* mainOutputLayer = nullptr
        );
        std::vector<std::vector<double>> query(
            const std::vector<double>& inputs, 
            Layer* mainInputLayer, 
            const std::vector<Layer*>& mainOutputLayers = std::vector<Layer*>()
        );
        std::vector<std::vector<double>> query(
            const std::vector<std::vector<double>>& inputs, 
            const std::vector<Layer*>& mainInputLayers, 
            const std::vector<Layer*>& mainOutputLayers = std::vector<Layer*>()
        );

        void train(
            const std::vector<double>& correctOutputs, 
            Layer* mainOutputLayer = nullptr
        );
        void train(
            const std::vector<std::vector<double>>& correctOutputs, 
            const std::vector<Layer*>& mainOutputLayers = std::vector<Layer*>()
        );
        
        void train(Layer* mainOutputLayer = nullptr);
        void train(const std::vector<Layer*>& mainOutputLayers = std::vector<Layer*>());
    };

    template <typename LayerType>
    LayerType* Net::addLayer(LayerType* layer, Layer* prevLayer)
    {
        if (layer) {
            if (prevLayer) {
                layer->addPrevLayer(prevLayer);
                prevLayer->addNextLayer(layer);
            }

            layer->addNet(this);
            layers.push_back(layer);

            initialized = false;
        } else
            Debug::warning("1 argument is nullptr");

        return layer;
    }

    template <typename LayerType>
    LayerType* Net::addLayer(LayerType* layer, std::vector<Layer*> prevLayers)
    {
        if (layer) {
            if (!prevLayers.empty())
                for (auto* prevLayer : prevLayers)
                    if (prevLayer) {
                        layer->addPrevLayer(prevLayer);
                        prevLayer->addNextLayer(layer);
                    }

            layer->addNet(this);
            layers.push_back(layer);

            initialized = false;
        } else
            Debug::warning("1 argument is nullptr");

        return layer;
    }
}