#include "neural.h"

#include <cassert>
#include <cmath>

#include <functional>
#include <numeric>
#include <random>

namespace
{

template <typename T>
T sqr(T value) { return (value * value); }

}

namespace anns
{

Neuron::Neuron(Type type, NeuralNetwork* parent) :
    m_type(type),
    m_parent(parent),
    m_axon(this)
{
    assert(m_parent != nullptr && "NeuralNetwork of Neuron is NULL.");
}

void Neuron::addDendrite(Dendrite* dendrite)
{
    m_dendrites.push_back(dendrite);
}

Axon* Neuron::axon()
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
        m_axon.setValue(m_parent->activationFunction()(inputValue()));
        break;
    }
}

double Neuron::inputValue() const
{
    return std::accumulate(m_dendrites.cbegin(),
                           m_dendrites.cend(),
                           0.0,
                           [](double sum, Dendrite* d) { return sum + (d->value() * d->synapse()->weight()); });
}

void Neuron::backPropagation(double expected)
{
    switch (m_type)
    {
    case Type::Output:
        {
            double neuronDelta =  (m_parent->activationFunction().derivative(m_axon.value()))
                                * (expected - m_axon.value());
            for (Dendrite* dendrite : m_dendrites)
            {
                dendrite->synapse()->setNeuronDelta(neuronDelta);
            }
        }
        break;
    default:
        assert(false && "For another neuron types need use backPropagation(void).");
        break;
    }
}

void Neuron::backPropagation()
{
    switch (m_type)
    {
    case Type::Hidden:
        {
            double neuronDelta =  (m_parent->activationFunction().derivative(m_axon.value()))
                                * (std::accumulate(m_axon.synapses().cbegin(),
                                                   m_axon.synapses().cend(),
                                                   0.0,
                                                   [](double sum, const Synapse* ss) { return sum + (ss->weight() * ss->neuronDelta()); })
                                   );
            for (Dendrite* dendrite : m_dendrites)
            {
                dendrite->synapse()->setNeuronDelta(neuronDelta);
            }
        }
        // NB: break not required here.
    case Type::Input:
    case Type::Bias:
        for (Synapse* synapse : m_axon.synapses())
        {
            double gradient = m_axon.value() * synapse->neuronDelta();
            double weightDelta =  (m_parent->learningRate() * gradient)
                                + (m_parent->learningMoment() * synapse->previousWeightDelta());
            synapse->correctWeight(weightDelta);
        }
        break;
    default:
        assert(false && "For Output neurons need use backPropagation(double).");
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

Dendrite* Synapse::dendrite()
{
    return &m_dendrite;
}

Axon* Synapse::axon()
{
    return m_axon;
}

const Axon* Synapse::axon() const
{
    return m_axon;
}

void Synapse::bind(Neuron* neuron)
{
    setAxon(neuron->axon());
}

void Synapse::setAxon(Axon* axon)
{
    m_axon = axon;
    axon->addSynapse(this);
}

double Synapse::weight() const
{
    return m_weight;
}

void Synapse::setWeight(double weight)
{
    m_weight = weight;
}

double Synapse::neuronDelta() const
{
    return m_neuronDelta;
}

void Synapse::setNeuronDelta(double delta)
{
    m_neuronDelta = delta;
}

void Synapse::correctWeight(double delta)
{
    m_weightDelta = delta;
    m_weight += m_weightDelta;
}

double Synapse::previousWeightDelta() const
{
    return m_weightDelta;
}

Dendrite::Dendrite(Synapse* owner) :
    m_owner(owner)
{
    assert(m_owner != nullptr && "Dendrite owner is NULL.");
}

Synapse* Dendrite::synapse()
{
    return m_owner;
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
    assert(m_owner != nullptr && "Axon owner is NULL.");
}

Neuron* Axon::neuron()
{
    return m_owner;
}

const Neuron* Axon::neuron() const
{
    return m_owner;
}

void Axon::addSynapse(Synapse* synapse)
{
    m_synapses.push_back(synapse);
}

std::vector<Synapse*>& Axon::synapses()
{
    return m_synapses;
}

const std::vector<Synapse*>& Axon::synapses() const
{
    return m_synapses;
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

double Sigmoid::derivative(double value) const
{
    return (1.0 - value) * value;
}

HyperbolicTangent::HyperbolicTangent() :
    ActivationFunction(Type::tanh)
{

}

double HyperbolicTangent::calculate(double value) const
{
    return (std::expm1(2.0 * value) / (std::exp(2.0 * value) + 1.0));
}

double HyperbolicTangent::derivative(double value) const
{
    return (1.0 - ::sqr(value));
}

ErrorFunction::ErrorFunction(Type type) :
    m_type(type)
{

}

std::unique_ptr<ErrorFunction> ErrorFunction::create(Type type)
{
    std::unique_ptr<ErrorFunction> result;
    switch (type)
    {
    case Type::MSE:
        result.reset(new MseError());
        break;
    case Type::RootMSE:
        result.reset(new RootMseError());
        break;
    case Type::Arctan:
        result.reset(new ArctanError());
        break;
    default:
        assert(false && "Unknown type of Error function.");
        break;
    }
    return result;
}

ErrorFunction::Type ErrorFunction::type() const
{
    return m_type;
}

double ErrorFunction::operator() (const std::vector<double>& actual,
                                  const std::vector<double>& expected) const
{
    assert(!actual.empty() && "Calculate error function: empty input vectors.");
    assert(actual.size() == expected.size() && "Caluculate error function: actual values count != expected values count.");

    return calculate(actual, expected);
}

MseError::MseError() :
    MseError(Type::MSE)
{

}

MseError::MseError(Type type) :
    ErrorFunction(type)
{

}

double MseError::calculate(const std::vector<double>& actual,
                           const std::vector<double>& expected) const
{
    return std::inner_product(expected.cbegin(),
                              expected.cend(),
                              actual.cbegin(),
                              0.0,
                              std::plus<double>(),
                              [](double e, double a) { return ::sqr(e - a); }
                             ) / static_cast<double>(actual.size());
}

RootMseError::RootMseError() :
    MseError(Type::RootMSE)
{

}

double RootMseError::calculate(const std::vector<double>& actual,
                               const std::vector<double>& expected) const
{
    return std::sqrt(MseError::calculate(actual, expected));
}

ArctanError::ArctanError() :
    ErrorFunction(Type::Arctan)
{

}

double ArctanError::calculate(const std::vector<double>& actual,
                              const std::vector<double>& expected) const
{
    return std::inner_product(expected.cbegin(),
                              expected.cend(),
                              actual.cbegin(),
                              0.0,
                              std::plus<double>(),
                              [](double e, double a) { return ::sqr(std::atan(e - a)); }
                             ) / static_cast<double>(actual.size());
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

ErrorFunction::Type NeuralNetworkOptions::errorFunctionType() const
{
    return m_errorFunctionType;
}

void NeuralNetworkOptions::setErrorFunctionType(ErrorFunction::Type type)
{
    m_errorFunctionType = type;
}

double NeuralNetworkOptions::learningRate() const
{
    return m_learningRate;
}

void NeuralNetworkOptions::setLearningRate(double rate)
{
    m_learningRate = rate;
}

double NeuralNetworkOptions::learningMoment() const
{
    return m_learningMoment;
}

void NeuralNetworkOptions::setLearningMoment(double moment)
{
    m_learningMoment = moment;
}

NeuralNetwork NeuralNetwork::create(const NeuralNetworkOptions& options)
{
    NeuralNetwork result;
    result.m_activation = ActivationFunction::create(options.activationFunctionType());
    result.m_error = ErrorFunction::create(options.errorFunctionType());
    result.m_learningRate = options.learningRate();
    result.m_learningMoment = options.learningMoment();

    result.m_synapses.resize(synapsesCount(options));

    result.m_inputLayer.resize(options.inputNeuronsCount(),
                               Neuron(Neuron::Type::Input, &result));
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
    std::vector<Neuron>* previousLayer = &result.m_inputLayer;
    for (std::vector<Neuron>& nextLayer : result.m_hiddenLayers)
    {
        nextLayer.reserve(options.hiddenNeuronsOfLayerCount() + (options.hasBiasNeurons() ? 1 : 0));
        while (nextLayer.size() < options.hiddenNeuronsOfLayerCount())
        {
            Neuron hidden(Neuron::Type::Hidden, &result);

            for (Neuron& prev : *previousLayer)
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
            Neuron bias(Neuron::Type::Bias, &result);
            nextLayer.push_back(bias);
        }
        previousLayer = &nextLayer;
    }

    result.m_outputLayer.resize(options.outputNeuronsCount(),
                                Neuron(Neuron::Type::Output, &result));
    result.m_outputs.reserve(options.outputNeuronsCount());
    for (Neuron& output : result.m_outputLayer)
    {
        for (Neuron& prev : *previousLayer)
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

std::vector<double> NeuralNetwork::forwardPass(const std::vector<double>& inputs)
{
    assert(inputs.size() == m_inputLayer.size() && "Input values count != input layer neurons count.");

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

void NeuralNetwork::backPropagation(const std::vector<double>& expected)
{
    assert(!expected.empty() && "Back propagation function: empty expected vector.");
    assert(expected.size() == m_outputs.size() && "Expected values count != output layer neurons count.");

    std::vector<double>::const_reverse_iterator exp_it = expected.crbegin();
    std::vector<Neuron>::reverse_iterator it, end;

    for (it = m_outputLayer.rbegin(), end = m_outputLayer.rend(); it != end; ++it)
    {
        it->backPropagation(*(exp_it++));
    }

    for (auto layer_it = m_hiddenLayers.rbegin(), layer_end = m_hiddenLayers.rend(); layer_it != layer_end; ++layer_it)
    {
        for (it = layer_it->rbegin(), end = layer_it->rend(); it != end; ++it)
        {
            it->backPropagation();
        }
    }

    for (it = m_inputLayer.rbegin(), end = m_inputLayer.rend(); it != end; ++it)
    {
        it->backPropagation();
    }
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

double NeuralNetwork::error(const std::vector<double>& actual,
                            const std::vector<double>& expected) const
{
    return errorFunction()(actual, expected);
}

double NeuralNetwork::learningRate() const
{
    return m_learningRate;
}

double NeuralNetwork::learningMoment() const
{
    return m_learningMoment;
}

const ActivationFunction& NeuralNetwork::activationFunction() const
{
    return *m_activation;
}

const ErrorFunction& NeuralNetwork::errorFunction() const
{
    return *m_error;
}

} // anns
