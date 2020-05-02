
#include "ANNetwork.h"
#include "Layer.h"
#include "InputData.h"


AnnUtilities::ANNetwork::ANNetwork()
{
}


AnnUtilities::ANNetwork::~ANNetwork()
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

void AnnUtilities::ANNetwork::init()
{
	_inputLayer = new Layer(nullptr, _settings._inputSize, _settings._momentum, _settings._hiddenActivationFunction);
	Layer* lastLayer = _inputLayer;
	for (int i = 0; i < _settings._numberOfHiddenLayers; i++)
	{
		Layer* hiddenLayer = new Layer(lastLayer, _settings._hiddenSize, _settings._momentum, _settings._hiddenActivationFunction);
		lastLayer->_nextLayer = hiddenLayer;
		lastLayer = hiddenLayer;
	}
	_outputLayer = new Layer(lastLayer, _settings._outputSize, _settings._momentum, _settings._outputActivationFunction);
	lastLayer->_nextLayer = _outputLayer;
}

void AnnUtilities::ANNetwork::epoch(const InputData* const inputData, const int inputSize, const float learningRate)
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

void AnnUtilities::ANNetwork::update(const int batchSize)
{
	Layer* l = _outputLayer;
	while (l->_prevLayer != nullptr)
	{
		l->update(_settings._learningRate, batchSize);
		l = l->_prevLayer;
	}
}
