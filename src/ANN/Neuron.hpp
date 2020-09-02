#pragma once

#include "Debug.hpp"
#include "Connect.hpp"
#include "FunctionActiv.hpp"

namespace net
{
    class Net;
    class Layer;

    class Neuron
    {
        friend Layer;
    protected:
        Net* net;
        Layer* layer;

        std::vector<Connect*> prevConnects;
        std::vector<Connect*> nextConnects;
    protected:
        FunctionActiv activFunct;
        double learningRate;

        double result;
        double error;
    protected:
        virtual void init() {};
        virtual void clear() {};
    protected:
        virtual void algorithmNeuron();
        virtual double algorithmTrain(Neuron* prev);
    public:
        Neuron(double learningRate = 0.0, const FunctionActiv& activFunct = FunctionActiv());
        virtual ~Neuron();

        virtual const std::vector<Connect*> getPrevConnects() const;
        virtual const std::vector<Connect*> getNextConnects() const;

        virtual const FunctionActiv& getActivationFunction() const;
        virtual double getLearningRate() const;

        virtual double getResult() const;
        virtual double getError() const;

        virtual void setActivationFunction(const FunctionActiv& activationFunction);
        virtual void setLearningRate(double learningRate);

        void query();
        void train(bool isMain = false);
    };
}