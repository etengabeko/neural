#ifndef NEURAL_H
#define NEURAL_H

#include <memory>
#include <vector>

/**
  * @namespace anns
  * @brief Artificial neural networks
  */
namespace anns
{
class ActivationFunction;
class Axon;
class Dendrite;
class ErrorFunction;
class NeuralNetwork;
class NeuralNetworkOptions;
class Neuron;
class Synapse;

class Axon
{
public:
    Axon() = default;
    explicit Axon(Neuron* owner);

    Neuron* neuron();
    const Neuron* neuron() const;

    void addSynapse(Synapse* synapse);
    const std::vector<Synapse*>& synapses() const;

    void setValue(double value);
    double value() const;

private:
    Neuron* m_owner = nullptr;
    std::vector<Synapse*> m_synapses;

    double m_value = 0.0;

};

class Dendrite
{
public:
    explicit Dendrite(Synapse* owner);

    Synapse* synapse();
    const Synapse* synapse() const;

    double value() const;

private:
    Synapse* m_owner;

};

class Synapse
{
public:
    Synapse();

    Dendrite* dendrite();
    const Dendrite* dendrite() const;

    Axon* axon();
    const Axon* axon() const;

    void bind(Neuron* neuron);
    void setAxon(Axon* axon);

    double weight() const;
    void setWeight(double weight);

    double delta() const;
    void setDelta(double delta);

private:
    Dendrite m_dendrite;
    Axon* m_axon;

    double m_weight = 0.0;
    double m_delta = 0.0;

};

class Neuron
{
public:
    enum class Type
    {
        Input,
        Output,
        Hidden,
        Bias
    };

public:
    Neuron(Type type, NeuralNetwork* parent);

    void addDendrite(Dendrite* dendrite);

    Axon* axon();

    void activate();

    void backPropagation(double expected);

private:
    double inputValue() const;

private:
    Type m_type;
    NeuralNetwork* m_parent;

    std::vector<Dendrite*> m_dendrites;
    Axon m_axon;

};

class ActivationFunction
{
public:
    enum class Type
    {
        sigm,
        tanh
    };

    explicit ActivationFunction(Type type);
    virtual ~ActivationFunction() = default;

    static std::unique_ptr<ActivationFunction> create(Type type);

    Type type() const;

    double operator() (double value) const;
    virtual double derivative(double value) const = 0;

protected:
    virtual double calculate(double value) const = 0;

private:
    const Type m_type;

};

class Sigmoid final : public ActivationFunction
{
public:
    Sigmoid();

    virtual double derivative(double value) const override;

private:
    virtual double calculate(double value) const override;

};

class HyperbolicTangent final : public ActivationFunction
{
public:
    HyperbolicTangent();

    virtual double derivative(double value) const override;

private:
    virtual double calculate(double value) const override;

};

class ErrorFunction
{
public:
    enum class Type
    {
        MSE,
        RootMSE,
        Arctan
    };

    explicit ErrorFunction(Type type);
    virtual ~ErrorFunction() = default;

    static std::unique_ptr<ErrorFunction> create(Type type);

    Type type() const;

    double operator() (const std::vector<double>& actual,
                       const std::vector<double>& expected) const;

protected:
    virtual double calculate(const std::vector<double>& actual,
                             const std::vector<double>& expected) const = 0;

private:
    const Type m_type;

};

class MseError : public ErrorFunction
{
public:
    MseError();

protected:
    explicit MseError(Type type);

    virtual double calculate(const std::vector<double>& actual,
                             const std::vector<double>& expected) const override;

};

class RootMseError final : public MseError
{
public:
    RootMseError();

private:
    virtual double calculate(const std::vector<double>& actual,
                             const std::vector<double>& expected) const override;

};

class ArctanError final : public ErrorFunction
{
public:
    ArctanError();

private:
    virtual double calculate(const std::vector<double>& actual,
                             const std::vector<double>& expected) const override;

};

class NeuralNetworkOptions
{
public:
    uint inputNeuronsCount() const;
    void setInputNeuronsCount(uint count);

    uint outputNeuronsCount() const;
    void setOutputNeuronsCount(uint count);

    uint hiddenLayersCount() const;
    void setHiddenLayersCount(uint count);

    uint hiddenNeuronsOfLayerCount() const;
    void setHiddenNeuronsOfLayerCount(uint count);

    bool hasBiasNeurons() const;
    void setHasBiasNeurons(bool hasBias);

    ActivationFunction::Type activationFunctionType() const;
    void setActivationFunctionType(ActivationFunction::Type type);

    ErrorFunction::Type errorFunctionType() const;
    void setErrorFunctionType(ErrorFunction::Type type);

    double learningRate() const;
    void setLearningRate(double rate);

    double learningMoment() const;
    void setLearningMoment(double moment);

private:
    uint m_inputNeuronsCount = 0;
    uint m_outputNeuronsCount = 0;
    uint m_hiddenLayersCount = 0;
    uint m_hiddenNeuronsOfLayerCount = 0;
    bool m_hasBiasNeurons = false;

    ActivationFunction::Type m_activationFunctionType = ActivationFunction::Type::sigm;
    ErrorFunction::Type m_errorFunctionType = ErrorFunction::Type::MSE;

    double m_learningRate = 0.0;
    double m_learningMoment = 0.0;

};

class NeuralNetwork
{
private:
    NeuralNetwork() = default;

public:
    NeuralNetwork(NeuralNetwork&& other) = default;
    NeuralNetwork& operator= (NeuralNetwork&& rhs) = default;

    static NeuralNetwork create(const NeuralNetworkOptions& options);

    std::vector<Synapse>* synapses() { return &m_synapses; } // FIXME

    std::vector<double> forwardPass(const std::vector<double>& inputs);
    void backPropagation(const std::vector<double>& expected);

    double error(const std::vector<double>& actual,
                 const std::vector<double>& expected) const;

    double learningRate() const;
    double learningMoment() const;

    const ActivationFunction& activationFunction() const;
    const ErrorFunction& errorFunction() const;

private:
    static uint synapsesCount(const NeuralNetworkOptions& options);
    void randomizeWeights();
    void activate();

private:
    std::unique_ptr<ActivationFunction> m_activation;
    std::unique_ptr<ErrorFunction> m_error;

    std::vector<Neuron> m_inputLayer;
    std::vector<std::vector<Neuron>> m_hiddenLayers;
    std::vector<Neuron> m_outputLayer;

    std::vector<Synapse> m_synapses;

    std::vector<Axon> m_inputs;
    std::vector<const Axon*> m_outputs;

    double m_learningRate = 0.0;
    double m_learningMoment = 0.0;

};

} // anns

#endif // NEURAL_H
