#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <neural.h>

using namespace anns;

int main(int /*argc*/, char* /*argv*/[])
{
    auto print_func = [](const std::vector<double>& values)
                      {
                          std::string res;
                          for (auto each : values)
                          {
                              if (!res.empty())
                                  res += " ";
                              res += std::to_string(each);
                          }
                          return res;
                      };

    NeuralNetworkOptions opt;
    opt.setInputNeuronsCount(2);
    opt.setOutputNeuronsCount(1);
    opt.setHiddenLayersCount(1);
    opt.setHiddenNeuronsOfLayerCount(4);
    opt.setHasBiasNeurons(true);
    opt.setActivationFunctionType(ActivationFunction::Type::sigm);
    opt.setErrorFunctionType(ErrorFunction::Type::MSE);
    opt.setLearningRate(0.7);
    opt.setLearningMoment(0.3);

    NeuralNetwork nnet = NeuralNetwork::create(opt);

    std::vector<std::pair<std::vector<double>, double>> inputs
    {
        { { 0, 0 }, 0.0 },
        { { 1, 0 }, 1.0 },
        { { 0, 1 }, 1.0 },
        { { 1, 1 }, 0.0 }
    };

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 3);

    double sumError = 0.0;
    size_t iteraton = 0;
    do
    {
        int randomIndex = distribution(generator);
        ++iteraton;

        std::vector<double> outputs = nnet.forwardPass(inputs[randomIndex].first);
        double error = nnet.error(outputs, std::vector<double>{ inputs[randomIndex].second });
        sumError += error;

        std::cout << "iteration #" << std::to_string(iteraton) << "\n";
        std::cout << "input set:  [" << print_func(inputs[randomIndex].first)  << "]\n";
        std::cout << "output set: [" << print_func(outputs) << "]\n";
        std::cout << "error:       " << std::to_string(error) << "\n";
        std::cout << "sum error:   " << std::to_string(sumError/iteraton) << "\n\n";
//        std::cout << std::to_string(sumError/iteraton) << "\n";

        nnet.backPropagation(std::vector<double>{ inputs[randomIndex].second });
    }
    while (sumError/iteraton > 0.01);

    return 0;
}
