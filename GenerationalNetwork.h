#include <cstdlib>
#include <ctime>
#include <vector>
#include "NeuralNetwork.h"
#pragma once
class GenerationalNetwork
{
private:
	int network_count_;
	std::vector<NeuralNetwork> networks_;
public:
	GenerationalNetwork(unsigned int seed,
						int networkCount,
						int inputSize,
						int hiddenLayerLenght,
						int * hiddenLayers, int outputSize);
	~GenerationalNetwork();
	int GetBestNetworkIndex();
	void RateNeuralNetworks	(double * expected, double * input);
	void RateNeuralNetworks	(double ** expected, double ** input, int dataSize);
	void SaveBestNetwork	(std::string fileName = "net.txt");
	void LoadBestNetwork	(std::string fileName = "net.txt");
	double GetAverageCost	(double * expected, double * input);
	double GetAverageCost	(double ** expected, double ** input, int dataSize);
	double GetLowestCost	(double * expected, double * input);
	double GetLowestCost	(double ** expected, double ** input, int dataSize);
	void NextGeneration(double rate = 0.01,
						int weightChangeProcent = 100,
						int biasChangeProcent = 100,
						int subWeightChangeProcent = 100,
						int subBiasChangeProcent = 100);
	NeuralNetwork GetNetwork(int index);
};
