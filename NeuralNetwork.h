#pragma once
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>

class NeuralNetwork
{
	double *	_inputLayer;	int _inputSize;
	double *	_output;		int _outputSize;
	double **	_hiddenLayer;	int * _hiddenLayerSize; int _hiddenLayerLenght;
	double **	_bias;			int _biasLenght;	int * _biasSizes;
	double **	_weights;		int _weightLenght;	int * _weightSizes;
	double Sigmoid(double value);
	double RandomDoubleRange(double min, double max);
	double RandomIntRange(int min, int max);
public:
	bool		best = false;
	NeuralNetwork(int inputLayerSize, int hiddenLayerLenght, int * hiddenLayers, int outputSize);
	void Remake(int inputLayerSize, int hiddenLayerLenght, int * hiddenLayers, int outputSize);
	NeuralNetwork();
	~NeuralNetwork();
	void ApplyDataFromOtherNetwork(NeuralNetwork network);
	int GetInputSize();
	int GetOutputSize();
	int GetHiddenLayerSize();
	int GetHiddenLayerSizeAt(int index);
	double GetBias(int x, int y);
	double GetWeight(int x, int y);
	void RandomChange(double rate = 0.01,
					  int weightChangeProcent = 100,
					  int biasChangeProcent = 100,
					  int subWeightChangeProcent = 100,
					  int subBiasChangeProcent = 100);
	void Export(std::string fileName = "net.txt");
	void Import(std::string fileName = "net.txt");
	void FirstRandomInit();
	double * GetOutput(double * input);
	double Cost(double ** expected, double ** input, int dataSize);
	double Cost(double *  expected,	double *  input);
};

