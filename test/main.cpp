#include <iostream>
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
    opt.setHiddenNeuronsOfLayerCount(2);
    opt.setHasBiasNeurons(false);
    opt.setActivationFunctionType(ActivationFunction::Type::sigm);

    NeuralNetwork nnet = NeuralNetwork::create(opt);

    std::vector<std::vector<double>> inputs{
                                               { 0, 0 },
                                               { 1, 0 },
                                               { 0, 1 },
                                               { 1, 1 }
                                           };

    for (const std::vector<double>& each : inputs)
    {
        std::vector<double> outputs = nnet.training(each);
        std::cout << "input set:  [" << print_func(each)  << "]\n";
        std::cout << "output set: [" << print_func(outputs) << "]\n";
    }

    return 0;
}
