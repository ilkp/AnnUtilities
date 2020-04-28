
#include "Network.h"
#include "Layer.h"
#include "InputData.h"


AnnUtilities::Network::Network()
{
}


AnnUtilities::Network::~Network()
{
	Clean();
}

void AnnUtilities::Network::Init(const int inputSize, const int hiddenSize, const int outputSize, const int hiddenLayers, AnnUtilities::ACTFUNC actfuncHidden, AnnUtilities::ACTFUNC actfuncOutput)
{
	_inputLayer = new Layer(nullptr, inputSize, actfuncHidden);
	Layer* lastLayer = _inputLayer;
	for (int i = 0; i < hiddenLayers; i++)
	{
		Layer* hiddenLayer = new Layer(lastLayer, hiddenSize, actfuncHidden);
		lastLayer->_nextLayer = hiddenLayer;
		lastLayer = hiddenLayer;
	}
	_outputLayer = new Layer(lastLayer, outputSize, actfuncOutput);
	lastLayer->_nextLayer = _outputLayer;
}

void AnnUtilities::Network::Epoch(const InputData* const inputData, const int inputSize, const float learningRate)
{
	Layer* l;
	for (int i = 0; i < inputSize; ++i)
	{
		_inputLayer->setOutputs(inputData[i]._input);
		propagateForward();
		propagateBackward(inputData[i]._label);
	}
	l = _outputLayer;
	while (l->_prevLayer != nullptr)
	{
		l->update(learningRate, inputSize);
		l = l->_prevLayer;
	}
}

float* AnnUtilities::Network::Test(const float* const inputData)
{
	_inputLayer->setOutputs(inputData);
	propagateForward();
	return _outputLayer->getOutput();
}

void AnnUtilities::Network::propagateForward()
{
	Layer* l = _inputLayer->_nextLayer;
	while (l != nullptr)
	{
		l->propagateForward();
		l = l->_nextLayer;
	}
}

void AnnUtilities::Network::propagateBackward(const float* const labels)
{
	Layer* l = _outputLayer;
	l->propagateBackward(labels);
	l = l->_prevLayer;

	while (l->_prevLayer != nullptr)
	{
		l->propagateBackward();
		l = l->_prevLayer;
	}
}

void AnnUtilities::Network::update(const int batchSize, const float learningRate)
{
	Layer* l = _outputLayer;
	while (l->_prevLayer != nullptr)
	{
		l->update(learningRate, batchSize);
		l = l->_prevLayer;
	}
}

void AnnUtilities::Network::Clean()
{
	if (!_outputLayer)
	{
		return;
	}

	Layer* l = _outputLayer->_prevLayer;
	while (l->_prevLayer != nullptr)
	{
		delete(l->_nextLayer);
		l = l->_prevLayer;
	}
	delete(l);
}