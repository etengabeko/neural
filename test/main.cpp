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

    // FIXME
    std::vector<Synapse>* ss = nnet.synapses();
    (*ss)[2].setWeight(0.45);
    (*ss)[3].setWeight(-0.12);
    (*ss)[4].setWeight(0.78);
    (*ss)[5].setWeight(0.13);
    (*ss)[6].setWeight(1.5);
    (*ss)[7].setWeight(-2.3);
    // -----

    std::vector<std::pair<std::vector<double>, double>> inputs
    {
        { { 0, 0 }, 0.0 },
        { { 1, 0 }, 1.0 },
        { { 0, 1 }, 1.0 },
        { { 1, 1 }, 0.0 }
    };

    for (const std::pair<std::vector<double>, double>& each : inputs)
    {
        std::vector<double> outputs = nnet.forwardPass(each.first);
        double error = nnet.error(outputs, std::vector<double>{ each.second });

        std::cout << "input set:  [" << print_func(each.first)  << "]\n";
        std::cout << "output set: [" << print_func(outputs) << "]\n";
        std::cout << "error:       " << std::to_string(error) << "\n\n";
    }

    return 0;
}
