
#include "ANNetwork.h"
#include "Layer.h"
#include "InputData.h"


AnnUtilities::ANNetwork::ANNetwork()
{
}


AnnUtilities::ANNetwork::~ANNetwork()
{
	Clean();
}

void AnnUtilities::ANNetwork::Init(const int inputSize, const int hiddenSize, const int outputSize, const int hiddenLayers, AnnUtilities::ACTFUNC actfuncHidden, AnnUtilities::ACTFUNC actfuncOutput)
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

void AnnUtilities::ANNetwork::Epoch(const InputData* const inputData, const int inputSize, const float learningRate)
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

float* AnnUtilities::ANNetwork::Test(const float* const inputData)
{
	_inputLayer->setOutputs(inputData);
	propagateForward();
	return _outputLayer->getOutput();
}

void AnnUtilities::ANNetwork::propagateForward()
{
	Layer* l = _inputLayer->_nextLayer;
	while (l != nullptr)
	{
		l->propagateForward();
		l = l->_nextLayer;
	}
}

void AnnUtilities::ANNetwork::propagateBackward(const float* const labels)
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

void AnnUtilities::ANNetwork::update(const int batchSize, const float learningRate)
{
	Layer* l = _outputLayer;
	while (l->_prevLayer != nullptr)
	{
		l->update(learningRate, batchSize);
		l = l->_prevLayer;
	}
}

void AnnUtilities::ANNetwork::Clean()
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