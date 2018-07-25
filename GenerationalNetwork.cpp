#include "stdafx.h"
#include "GenerationalNetwork.h"

GenerationalNetwork::GenerationalNetwork(unsigned int seed, int networkCount, int inputSize, int hiddenLayerLenght, int * hiddenLayers, int outputSize)
{
	srand(seed);
	network_count_ = networkCount;
	networks_ = std::vector<NeuralNetwork>(network_count_);
	for (int i = 0; i < network_count_; i++)
	{
		networks_[i].Remake(inputSize, hiddenLayerLenght, hiddenLayers, outputSize);
	}
}


GenerationalNetwork::~GenerationalNetwork()
{
	networks_.clear();
}

int GenerationalNetwork::GetBestNetworkIndex()
{
	for (auto i = 0; i < network_count_; i++)
	{
		if (networks_.at(i).best) return i;
	}
	return NULL;
}

void GenerationalNetwork::RateNeuralNetworks(double * expected, double * input)
{
	const auto errors = new double[network_count_];

	for (auto i = 0; i < network_count_; i++)
	{
		networks_.at(i).best = false;
	}

	for (auto i = 0; i < network_count_; i++)
	{
		errors[i] = networks_[i].Cost(expected, input);
	}
	auto low = errors[0];
	auto index = 0;
	for (auto i = 1; i < network_count_; i++)
	{
		if (errors[i] <= low) { low = errors[i]; index = i; }
	}
	networks_.at(index).best = true;
	delete[] errors;
}

void GenerationalNetwork::RateNeuralNetworks(double ** expected, double ** input, int dataSize)
{
	const auto errors = new double[network_count_];

	for (auto i = 0; i < network_count_; i++)
	{
		networks_.at(i).best = false;
	}

	for (auto i = 0; i < network_count_; i++)
	{
		errors[i] = networks_[i].Cost(expected, input, dataSize);
	}
	auto low = errors[0];
	auto index = 0;
	for (auto i = 1; i < network_count_; i++)
	{
		if (errors[i] <= low) { low = errors[i]; index = i; }
	}
	networks_.at(index).best = true;
	delete[] errors;
}

void GenerationalNetwork::SaveBestNetwork(std::string fileName)
{
	for (auto i = 0; i < network_count_; i++)
	{
		if (networks_.at(i).best)
		{
			networks_.at(i).Export(fileName);
			return;
		}
	}
	networks_[0].Export(fileName);
}

void GenerationalNetwork::LoadBestNetwork(std::string fileName)
{
	for (auto i = 0; i < network_count_; i++)
	{
		if (networks_[i].best)
		{
			networks_[i].Import(fileName);
			return;
		}
	}
	networks_[0].Import(fileName);
}

double GenerationalNetwork::GetAverageCost(double * expected, double * input)
{
	double cost = 0;

	for (int i = 0; i < network_count_; i++)
	{
		cost += networks_[i].Cost(expected, input);
	}

	return cost / network_count_;
}

double GenerationalNetwork::GetAverageCost(double ** expected, double ** input, const int dataSize)
{
	double cost = 0;

	for (auto i = 0; i < network_count_; i++)
	{
		cost += networks_[i].Cost(expected, input, dataSize);
	}

	return cost / network_count_;
}

double GenerationalNetwork::GetLowestCost(double * expected, double * input)
{
	for (auto i = 0; i < network_count_; i++)
	{
		if (networks_.at(i).best) return networks_.at(i).Cost(expected, input);
	}
	return -1;
}

double GenerationalNetwork::GetLowestCost(double ** expected, double ** input, int dataSize)
{
	for (auto i = 0; i < network_count_; i++)
	{
		if (networks_.at(i).best) return networks_.at(i).Cost(expected, input, dataSize);
	}
	return -1;
}

void GenerationalNetwork::NextGeneration(double rate, int weightChangeProcent, int biasChangeProcent,
	int subWeightChangeProcent, int subBiasChangeProcent)
{
	const auto best_index = GetBestNetworkIndex();
	for(auto i = 0;i < network_count_;i++)
	{
		if(networks_.at(i).best) continue;
		networks_.at(i).ApplyDataFromOtherNetwork(networks_[best_index]);
		networks_.at(i).RandomChange(rate, weightChangeProcent,biasChangeProcent,subWeightChangeProcent,subBiasChangeProcent);
	}
}

NeuralNetwork GenerationalNetwork::GetNetwork(int index)
{
	return networks_.at(index);
}
