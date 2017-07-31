#include "neural.h"

#include <cassert>
#include <cmath>

#include <algorithm>
#include <random>

namespace anns
{

Neuron::Neuron(Type type) :
    m_type(type),
    m_axon(this)
{

}

void Neuron::setActivationFunction(ActivationFunction* func)
{
    m_activation = func;
}

void Neuron::addDendrite(const Dendrite* dendrite)
{
    m_dendrites.push_back(dendrite);
}

const Axon* Neuron::axon() const
{
    return &m_axon;
}

void Neuron::activate()
{
    switch (m_type)
    {
    case Type::Input:
        m_axon.setValue(m_dendrites.front()->value());
        break;
    case Type::Bias:
        m_axon.setValue(1.0);
        break;
    case Type::Output:
    case Type::Hidden:
        double value = std::accumulate(m_dendrites.cbegin(),
                                       m_dendrites.cend(),
                                       0.0,
                                       [](double sum, const Dendrite* d) { return sum + (d->value() * d->synapse()->weight()); });
        m_axon.setValue((*m_activation)(value));
        break;
    }
}

Synapse::Synapse() :
    m_dendrite(this),
    m_axon(nullptr)
{

}

const Dendrite* Synapse::dendrite() const
{
    return &m_dendrite;
}

const Axon* Synapse::axon() const
{
    return m_axon;
}

void Synapse::bind(const Neuron* neuron)
{
    setAxon(neuron->axon());
}

void Synapse::setAxon(const Axon* axon)
{
    m_axon = axon;
}

double Synapse::weight() const
{
    return m_weight;
}

void Synapse::setWeight(double weight)
{
    m_weight = weight;
}

Dendrite::Dendrite(Synapse* owner) :
    m_owner(owner)
{
    assert(m_owner != nullptr && "Dendrite owner is NULL");
}

const Synapse* Dendrite::synapse() const
{
    return m_owner;
}

double Dendrite::value() const
{
    return synapse()->axon()->value();
}

Axon::Axon(Neuron* owner) :
    m_owner(owner)
{
    assert(m_owner != nullptr && "Axon owner is NULL");
}

Neuron* Axon::neuron()
{
    return m_owner;
}

void Axon::setValue(double value)
{
    m_value = value;
}

double Axon::value() const
{
    return m_value;
}

ActivationFunction::ActivationFunction(Type type) :
    m_type(type)
{

}

double ActivationFunction::operator() (double value) const
{
    return calculate(value);
}

ActivationFunction::Type ActivationFunction::type() const
{
    return m_type;
}

std::unique_ptr<ActivationFunction> ActivationFunction::create(Type type)
{
    std::unique_ptr<ActivationFunction> result;
    switch (type)
    {
    case Type::sigm:
        result.reset(new Sigmoid());
        break;
    case Type::tanh:
        result.reset(new HyperbolicTangent());
        break;
    default:
        assert(false && "Unknown type of Activation function.");
        break;
    }
    return result;
}

Sigmoid::Sigmoid() :
    ActivationFunction(Type::sigm)
{

}

double Sigmoid::calculate(double value) const
{
    return (1.0 / (1.0 + std::exp(-value)));
}

HyperbolicTangent::HyperbolicTangent() :
    ActivationFunction(Type::tanh)
{

}

double HyperbolicTangent::calculate(double value) const
{
    return (std::exp(2.0 * value) - 1.0) / (std::exp(2.0 * value) + 1.0);
}

uint NeuralNetworkOptions::inputNeuronsCount() const
{
    return m_inputNeuronsCount;
}

void NeuralNetworkOptions::setInputNeuronsCount(uint count)
{
    m_inputNeuronsCount = count;
}

uint NeuralNetworkOptions::outputNeuronsCount() const
{
    return m_outputNeuronsCount;
}

void NeuralNetworkOptions::setOutputNeuronsCount(uint count)
{
    m_outputNeuronsCount = count;
}

uint NeuralNetworkOptions::hiddenLayersCount() const
{
    return m_hiddenLayersCount;
}

void NeuralNetworkOptions::setHiddenLayersCount(uint count)
{
    m_hiddenLayersCount = count;
}

uint NeuralNetworkOptions::hiddenNeuronsOfLayerCount() const
{
    return m_hiddenNeuronsOfLayerCount;
}

void NeuralNetworkOptions::setHiddenNeuronsOfLayerCount(uint count)
{
    m_hiddenNeuronsOfLayerCount = count;
}

bool NeuralNetworkOptions::hasBiasNeurons() const
{
    return m_hasBiasNeurons;
}

void NeuralNetworkOptions::setHasBiasNeurons(bool hasBias)
{
    m_hasBiasNeurons = hasBias;
}

ActivationFunction::Type NeuralNetworkOptions::activationFunctionType() const
{
    return m_activationFunctionType;
}

void NeuralNetworkOptions::setActivationFunctionType(ActivationFunction::Type type)
{
    m_activationFunctionType = type;
}

NeuralNetwork NeuralNetwork::create(const NeuralNetworkOptions& options)
{
    NeuralNetwork result;
    result.m_activation = ActivationFunction::create(options.activationFunctionType());

    result.m_synapses.resize(synapsesCount(options));

    result.m_inputLayer.resize(options.inputNeuronsCount(), Neuron(Neuron::Type::Input));
    result.m_inputs.resize(options.inputNeuronsCount());
    size_t processedSynapses = 0;
    for (Neuron& input : result.m_inputLayer)
    {
        Synapse& nextSynapse = result.m_synapses[processedSynapses];
        input.addDendrite(nextSynapse.dendrite());

        nextSynapse.setAxon(&result.m_inputs[processedSynapses]);

        ++processedSynapses;
    }

    result.m_hiddenLayers.resize(options.hiddenLayersCount());
    const std::vector<Neuron>* previousLayer = &result.m_inputLayer;
    for (std::vector<Neuron>& nextLayer : result.m_hiddenLayers)
    {
        nextLayer.reserve(options.hiddenNeuronsOfLayerCount() + (options.hasBiasNeurons() ? 1 : 0));
        while (nextLayer.size() < options.hiddenNeuronsOfLayerCount())
        {
            Neuron hidden(Neuron::Type::Hidden);
            hidden.setActivationFunction(result.m_activation.get());

            for (const Neuron& prev : *previousLayer)
            {
                Synapse& nextSynapse = result.m_synapses[processedSynapses];
                hidden.addDendrite(nextSynapse.dendrite());
                nextSynapse.bind(&prev);

                ++processedSynapses;
            }
            nextLayer.push_back(hidden);
        }
        if (options.hasBiasNeurons())
        {
            Neuron bias(Neuron::Type::Bias);
            nextLayer.push_back(bias);
        }
        previousLayer = &nextLayer;
    }

    result.m_outputLayer.resize(options.outputNeuronsCount(), Neuron(Neuron::Type::Output));
    result.m_outputs.reserve(options.outputNeuronsCount());
    for (Neuron& output : result.m_outputLayer)
    {
        output.setActivationFunction(result.m_activation.get());

        for (const Neuron& prev : *previousLayer)
        {
            Synapse& nextSynapse = result.m_synapses[processedSynapses];
            output.addDendrite(nextSynapse.dendrite());
            nextSynapse.bind(&prev);

            ++processedSynapses;
        }
        result.m_outputs.push_back(output.axon());
    }

    result.randomizeWeights();

    return result;
}

uint NeuralNetwork::synapsesCount(const NeuralNetworkOptions& options)
{
    const uint biasNeuronsOfLayer = options.hasBiasNeurons() ? 1 : 0;

    uint result = options.inputNeuronsCount();
    result += options.inputNeuronsCount() * options.hiddenNeuronsOfLayerCount();
    result += (options.hiddenLayersCount() - 1) * (options.hiddenNeuronsOfLayerCount() * (options.hiddenNeuronsOfLayerCount() + biasNeuronsOfLayer));
    result += options.outputNeuronsCount() * (options.hiddenNeuronsOfLayerCount() + biasNeuronsOfLayer);

    return result;
}

void NeuralNetwork::randomizeWeights()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (Synapse& each : m_synapses)
    {
        each.setWeight(distribution(generator));
    }
}

std::vector<double> NeuralNetwork::training(const std::vector<double>& inputs)
{
    assert(inputs.size() == m_inputLayer.size() && "input values count != input layer neurons count");

    {
        std::vector<double>::const_iterator it = inputs.cbegin();
        for (Axon& input : m_inputs)
        {
            input.setValue(*(it++));
        }
    }

    activate();

    std::vector<double> result(m_outputs.size());
    std::vector<double>::iterator it = result.begin();
    for (const Axon* output : m_outputs)
    {
        *(it++) = output->value();
    }

    return result;
}

void NeuralNetwork::activate()
{
    for (Neuron& each : m_inputLayer)
    {
        each.activate();
    }

    for (std::vector<Neuron>& layer : m_hiddenLayers)
    {
        for (Neuron& each : layer)
        {
            each.activate();
        }
    }

    for (Neuron& each : m_outputLayer)
    {
        each.activate();
    }
}

} // anns
