#include <random>
#include "Layer.h"

AnnUtilities::Layer::Layer(Layer* previousLayer, int layerSize, float(*activationFunction)(float), float(*derivativeFunction)(float))
	: _prevLayer(previousLayer), _layerSize(layerSize), _activationFunction(activationFunction), _derivativeFunction(derivativeFunction)
{
	_outputs = new float[layerSize]();

	if (previousLayer != nullptr)
	{
		_deltaWeights = new float[layerSize * previousLayer->_layerSize];
		_weights = new float[layerSize * previousLayer->_layerSize];
		_deltaBiases = new float[layerSize];
		_biases = new float[layerSize];
		_error = new float[layerSize];
		for (int i = 0; i < layerSize; ++i)
		{
			_biases[i] = 1.5f * (float(rand()) / float(RAND_MAX)) - 0.75f;
			_deltaBiases[i] = 0.0f;
			_error[i] = 0.0f;
			for (int j = 0; j < previousLayer->_layerSize; ++j)
			{
				_weights[i * _prevLayer->_layerSize + j] = 1.5f * (float(rand()) / float(RAND_MAX)) - 0.75f;
				_deltaWeights[i * _prevLayer->_layerSize + j] = 0.0f;
			}
		}
	}
}

AnnUtilities::Layer::~Layer()
{
	delete[](_outputs);
	delete[](_error);
	delete[](_weights);
	delete[](_deltaWeights);
	delete[](_biases);
	delete[](_deltaBiases);
}

// Calculates output values for each node
void AnnUtilities::Layer::propagationForward()
{
	for (int i = 0; i < _layerSize; ++i)
	{
		_outputs[i] = 0.0f;
		for (int j = 0; j < _prevLayer->_layerSize; ++j)
		{
			_outputs[i] += _weights[i * _prevLayer->_layerSize + j] * _prevLayer->_outputs[j];
		}
		_outputs[i] += _biases[i];
		_outputs[i] = _activationFunction(_outputs[i]);
	}
}

// Calculates error value for each node
void AnnUtilities::Layer::propagationBackward()
{
	for (int i = 0; i < _layerSize; ++i)
	{
		_error[i] = 0.0f;
	}
	for (int i = 0; i < _nextLayer->_layerSize; ++i)
	{
		for (int j = 0; j < _layerSize; ++j)
		{
			_error[j] += _nextLayer->_error[i] * _nextLayer->_weights[i * _layerSize + j];
		}
	}
	for (int i = 0; i < _layerSize; ++i)
	{
		_error[i] = _derivativeFunction(_outputs[i]) * _error[i];
	}
}

void AnnUtilities::Layer::calculateDelta()
{
	for (int i = 0; i < _layerSize; ++i)
	{
		_deltaBiases[i] += _error[i];
		for (int j = 0; j < _prevLayer->_layerSize; ++j)
		{
			_deltaWeights[i * _prevLayer->_layerSize + j] += _error[i] * _prevLayer->_outputs[j];
		}
	}
}

void AnnUtilities::Layer::update(const float learningRate, const int epochs)
{
	for (int i = 0; i < _layerSize; ++i)
	{
		_biases[i] -= learningRate * _deltaBiases[i] / epochs;
		_deltaBiases[i] = 0.0f;
		for (int j = 0; j < _prevLayer->_layerSize; ++j)
		{
			_weights[i * _prevLayer->_layerSize + j] -= learningRate * _deltaWeights[i * _prevLayer->_layerSize + j] / epochs;
			_deltaWeights[i * _prevLayer->_layerSize + j] = 0.0f;
		}
	}
}