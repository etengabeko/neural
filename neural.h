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
class Neuron;
class Synapse;

class Axon
{
public:
    Axon() = default;
    explicit Axon(Neuron* owner);

    Neuron* neuron();

    void setValue(double value);
    double value() const;

private:
    Neuron* m_owner = nullptr;
    double m_value = 0.0;

};

class Dendrite
{
public:
    explicit Dendrite(Synapse* owner);

    const Synapse* synapse() const;

    double value() const;

private:
    Synapse* m_owner;

};

class Synapse
{
public:
    Synapse();

    const Dendrite* dendrite() const;
    const Axon* axon() const;

    void bind(const Neuron* neuron);
    void setAxon(const Axon* axon);

    double weight() const;
    void setWeight(double weight);

private:
    const Dendrite m_dendrite;
    const Axon* m_axon;

    double m_weight = 0.0;

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
    explicit Neuron(Type type);

    void setActivationFunction(ActivationFunction* func);

    void addDendrite(const Dendrite* dendrite);

    const Axon* axon() const;

    void activate();

private:
    Type m_type;
    ActivationFunction* m_activation = nullptr;

    std::vector<const Dendrite*> m_dendrites;
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

protected:
    virtual double calculate(double value) const = 0;

private:
    const Type m_type;

};

class Sigmoid final : public ActivationFunction
{
public:
    Sigmoid();

private:
    virtual double calculate(double value) const override;

};

class HyperbolicTangent final : public ActivationFunction
{
public:
    HyperbolicTangent();

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

private:
    uint m_inputNeuronsCount = 0;
    uint m_outputNeuronsCount = 0;
    uint m_hiddenLayersCount = 0;
    uint m_hiddenNeuronsOfLayerCount = 0;
    bool m_hasBiasNeurons = false;

    ActivationFunction::Type m_activationFunctionType = ActivationFunction::Type::sigm;

    // TODO

};

class NeuralNetwork
{
private:
    NeuralNetwork() = default;

public:
    NeuralNetwork(NeuralNetwork&& other) = default;
    NeuralNetwork& operator= (NeuralNetwork&& rhs) = default;

    static NeuralNetwork create(const NeuralNetworkOptions& options);

    std::vector<double> training(const std::vector<double>& inputs);

private:
    static uint synapsesCount(const NeuralNetworkOptions& options);
    void randomizeWeights();
    void activate();

private:
    std::unique_ptr<ActivationFunction> m_activation;

    std::vector<Neuron> m_inputLayer;
    std::vector<std::vector<Neuron>> m_hiddenLayers;
    std::vector<Neuron> m_outputLayer;

    std::vector<Synapse> m_synapses;

    std::vector<Axon> m_inputs;
    std::vector<const Axon*> m_outputs;

};

} // anns

#endif // NEURAL_H
