#include "stdafx.h"
#include "NeuralNetwork.h"


double NeuralNetwork::Sigmoid(const double value)
{
	return 1 / (1 + std::exp(-value));
}

double NeuralNetwork::RandomDoubleRange(const double min, const double max)
{
	const auto f = static_cast<double>(rand()) / RAND_MAX;
	return min + f * (max - min);
}

double NeuralNetwork::RandomIntRange(const int min, const int max)
{
	return (rand() % (max + 1)) + min;
}

NeuralNetwork::NeuralNetwork(int inputLayerSize, int hiddenLayerLenght, int * hiddenLayers, int outputSize)
{
	_inputSize = inputLayerSize;
	_outputSize = outputSize;
	_inputLayer = new double[inputLayerSize];
	_output = new double[outputSize];

	_hiddenLayerLenght = hiddenLayerLenght;
	_hiddenLayerSize = hiddenLayers;
	_hiddenLayer = new double*[hiddenLayerLenght];

	_biasLenght = hiddenLayerLenght;
	_biasSizes = new int[_biasLenght];

	for (auto i = 0; i < hiddenLayerLenght; i++)
	{
		_hiddenLayer[i] = new double[hiddenLayers[i]];
		_biasSizes[i] = hiddenLayers[i];
	}

	_weightLenght = hiddenLayerLenght + 1;
	_weightSizes = new int[_weightLenght];
	_weightSizes[0] = inputLayerSize * _hiddenLayerSize[0];
	_weightSizes[hiddenLayerLenght] = outputSize * _hiddenLayerSize[_hiddenLayerLenght - 1];

	for (auto i = 1; i < _weightLenght - 1; i++)
	{
		_weightSizes[i] = _hiddenLayerSize[i] * _hiddenLayerSize[i - 1];
	}

	_weights = new double*[_weightLenght];
	for (auto i = 0; i < _weightLenght; i++)
	{
		_weights[i] = new double[_weightSizes[i]];
	}

	_bias = new double*[_biasLenght];
	for (auto i = 0; i < _biasLenght; i++)
	{
		_bias[i] = new double[_biasSizes[i]];
	}

	FirstRandomInit();
}

void NeuralNetwork::Remake(int inputLayerSize, int hiddenLayerLenght, int * hiddenLayers, int outputSize)
{
	_inputSize = inputLayerSize;
	_outputSize = outputSize;
	_inputLayer = new double[inputLayerSize];
	_output = new double[outputSize];

	_hiddenLayerLenght = hiddenLayerLenght;
	_hiddenLayerSize = hiddenLayers;
	_hiddenLayer = new double*[hiddenLayerLenght];

	_biasLenght = hiddenLayerLenght;
	_biasSizes = new int[_biasLenght];

	for (auto i = 0; i < hiddenLayerLenght; i++)
	{
		_hiddenLayer[i] = new double[hiddenLayers[i]];
		_biasSizes[i] = hiddenLayers[i];
	}

	_weightLenght = hiddenLayerLenght + 1;
	_weightSizes = new int[_weightLenght];
	_weightSizes[0] = inputLayerSize * _hiddenLayerSize[0];
	_weightSizes[hiddenLayerLenght] = outputSize * _hiddenLayerSize[_hiddenLayerLenght - 1];

	for (auto i = 1; i < _weightLenght - 1; i++)
	{
		_weightSizes[i] = _hiddenLayerSize[i] * _hiddenLayerSize[i - 1];
	}

	_weights = new double*[_weightLenght];
	for (auto i = 0; i < _weightLenght; i++)
	{
		_weights[i] = new double[_weightSizes[i]];
	}

	_bias = new double*[_biasLenght];
	for (auto i = 0; i < _biasLenght; i++)
	{
		_bias[i] = new double[_biasSizes[i]];
	}

	FirstRandomInit();
}

NeuralNetwork::NeuralNetwork()
{}//Default Constructor


NeuralNetwork::~NeuralNetwork()
{
	//TODO: MEMORY FREE
}

void NeuralNetwork::ApplyDataFromOtherNetwork(NeuralNetwork network)
{
	for (int x = 0; x < _biasLenght; x++)
	{
		for (int y = 0; y < _biasSizes[x]; y++)
		{
			_bias[x][y] = network.GetBias(x, y);
		}
	}
	for (int x = 0; x < _weightLenght; x++)
	{
		for (int y = 0; y < _weightSizes[x]; y++)
		{
			_weights[x][y] = network.GetWeight(x, y);
		}
	}
}

int NeuralNetwork::GetInputSize()
{
	return _inputSize;
}

int NeuralNetwork::GetOutputSize()
{
	return _outputSize;
}

int NeuralNetwork::GetHiddenLayerSize()
{
	return _hiddenLayerLenght;
}

int NeuralNetwork::GetHiddenLayerSizeAt(int index)
{
	return _hiddenLayerSize[index];
}

double NeuralNetwork::GetBias(int x, int y)
{
	return _bias[x][y];
}

double NeuralNetwork::GetWeight(int x, int y)
{
	return _weights[x][y];
}

void NeuralNetwork::RandomChange(double rate, int weightChangeProcent, int biasChangeProcent, int subWeightChangeProcent, int subBiasChangeProcent)
{
	if (biasChangeProcent == 100 && subBiasChangeProcent == 100)
	{
		for (int i = 0; i < _biasLenght; i++)
		{
			for (int j = 0; j < _biasSizes[i]; j++)
			{
				_bias[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
			}
		}
	}
	else if (biasChangeProcent == 100 && subBiasChangeProcent != 100)
	{
		for (int i = 0; i < _biasLenght; i++)
		{
			for (int j = 0; j < _biasSizes[i]; j++)
			{
				if (RandomIntRange(1, 100) <= subBiasChangeProcent)
				{
					_bias[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
				}
			}
		}
	}
	else if (biasChangeProcent != 100 && subBiasChangeProcent != 100)
	{
		if (RandomIntRange(1, 100) <= biasChangeProcent)
		{
			for (int i = 0; i < _biasLenght; i++)
			{
				for (int j = 0; j < _biasSizes[i]; j++)
				{
					if (RandomIntRange(1, 100) <= subBiasChangeProcent)
					{
						_bias[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
					}
				}
			}
		}
	}
	else
	{
		if (RandomIntRange(1, 100) <= biasChangeProcent)
		{
			for (int i = 0; i < _biasLenght; i++)
			{
				for (int j = 0; j < _biasSizes[i]; j++)
				{
					_bias[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
				}
			}
		}
	}
	if (weightChangeProcent == 100 && subWeightChangeProcent == 100)
	{
		for (int i = 0; i < _weightLenght; i++)
		{
			for (int j = 0; j < _weightSizes[i]; j++)
			{
				_weights[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
			}
		}
	}
	else if (weightChangeProcent == 100 && subWeightChangeProcent != 100)
	{
		for (int i = 0; i < _weightLenght; i++)
		{
			for (int j = 0; j < _weightSizes[i]; j++)
			{
				if (RandomIntRange(1, 100) <= subWeightChangeProcent)
				{
					_weights[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
				}
			}
		}
	}
	else if (weightChangeProcent != 100 && subWeightChangeProcent != 100)
	{
		if (RandomIntRange(1, 100) <= weightChangeProcent)
		{
			for (int i = 0; i < _weightLenght; i++)
			{
				for (int j = 0; j < _weightSizes[i]; j++)
				{
					if (RandomIntRange(1, 100) <= subWeightChangeProcent)
					{
						_weights[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
					}
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < _weightLenght; i++)
		{
			for (int j = 0; j < _weightSizes[i]; j++)
			{
				if (RandomIntRange(1, 100) <= subWeightChangeProcent)
				{
					_weights[i][j] += RandomDoubleRange(-0.5, 0.5) * rate;
				}
			}
		}

	}
}

void NeuralNetwork::Export(std::string fileName)
{
	std::ofstream write(fileName);

	write << _inputSize << " " << _outputSize << " " << _hiddenLayerLenght << '\n';

	for (int i = 0; i < _hiddenLayerLenght; i++)
	{
		if (i + 1 != _hiddenLayerLenght)	write << _hiddenLayerSize[i] << ' ';
		else								write << _hiddenLayerSize[i] << '\n';
	}

	write << _biasLenght << '\n';

	for (int i = 0; i < _biasLenght; i++)
	{
		write << _biasSizes[i] << '\n';
		for (int t = 0; t < _biasSizes[i]; t++)
		{
			if (t + 1 != _biasSizes[i]) write << _bias[i][t] << ' ';
			else						write << _bias[i][t] << '\n';
		}
	}

	write << _weightLenght << '\n';
	for (int i = 0; i < _weightLenght; i++)
	{
		write << _weightSizes[i] << '\n';
		for (int t = 0; t < _weightSizes[i]; t++)
		{
			if (t + 1 != _weightSizes[i])	write << _weights[i][t] << ' ';
			else							write << _weights[i][t];
		}
		if (i + 1 != _weightLenght) write << '\n';
	}

	write.close();
}

void NeuralNetwork::Import(std::string fileName)
{
	std::ifstream read(fileName);

	read >> _inputSize >> _outputSize >> _hiddenLayerLenght;

	for (int i = 0; i < _hiddenLayerLenght; i++)	read >> _hiddenLayerSize[i];

	read >> _biasLenght;

	for (int i = 0; i < _biasLenght; i++)
	{
		read >> _biasSizes[i];
		for (int t = 0; t < _biasSizes[i]; t++)		read >> _bias[i][t];
	}

	read >> _weightLenght;

	for (int i = 0; i < _weightLenght; i++)
	{
		read >> _weightSizes[i];
		for (int t = 0; t < _weightSizes[i]; t++)	read >> _weights[i][t];
	}
	read.close();
}

void NeuralNetwork::FirstRandomInit()
{
	for (auto x = 0; x < _weightLenght; x++)
	{
		for (auto t = 0; t < _weightSizes[x]; t++)
		{
			_weights[x][t] = RandomDoubleRange(-1.5, 1.5);
		}
	}
	for (auto x = 0; x < _biasLenght; x++)
	{
		for (auto y = 0; y < _biasSizes[x]; y++)
		{
			_bias[x][y] = RandomDoubleRange(-2.5, 1);
		}
	}
}

double * NeuralNetwork::GetOutput(double * input)
{
	_inputLayer = input;
	double sum;

	for (int i = 0; i < _hiddenLayerSize[0]; i++)
	{
		sum = 0;
		for (int t = 0; t < _inputSize; t++)
		{
			sum += _inputLayer[t] * _weights[0][t];
		}
		sum += _bias[0][i];
		_hiddenLayer[0][i] = Sigmoid(sum);
	}

	for (int i = 1; i < _hiddenLayerLenght; i++)
	{
		sum = 0;
		for (int j = 0; j < _hiddenLayerSize[i]; j++)
		{
			for (int t = 0; t < _hiddenLayerSize[i - 1]; t++)
			{
				sum += _hiddenLayer[i - 1][t] * _weights[i][t];
			}
			sum += _bias[i][j];
			_hiddenLayer[i][j] = Sigmoid(sum);
		}
	}

	for (int i = 0; i < _outputSize; i++)
	{
		sum = 0;
		for (int j = 0; j < _hiddenLayerSize[_hiddenLayerLenght - 1]; j++)
		{
			sum += _hiddenLayer[_hiddenLayerLenght - 1][j] * _weights[_weightLenght - 1][j];
		}
		_output[i] = Sigmoid(sum);
	}

	return _output;
}

double NeuralNetwork::Cost(double ** expected, double ** input, int dataSize)
{
	double cost = 0;
	for (auto i = 0; i < dataSize; i++)
	{
		const auto value = GetOutput(input[i]);
		for (auto j = 0; j < _outputSize; j++)
		{
			cost += std::pow(expected[i][j] - value[j], 2);
		}
	}
	return cost;
}

double NeuralNetwork::Cost(double * expected, double * input)
{
	double cost = 0;
	auto value = GetOutput(input);
	for (int i = 0; i < _outputSize; i++)
	{
		cost += std::pow(expected[i] - value[i], 2);
	}
	return cost;
}
