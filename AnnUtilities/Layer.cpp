
#include <random>
#include "Layer.h"
#include "Functions.h"


AnnUtilities::Layer::Layer(Layer* previousLayer, int layerSize, float momentum, AnnUtilities::ACTFUNC actfunc) : _prevLayer(previousLayer), _layerSize(layerSize), _momentum(momentum), _actfunc(actfunc)
{
	_outputs = new float[layerSize];
	for (int i = 0; i < layerSize; ++i)
	{
		_outputs[i] = 0.0f;
	}

	if (previousLayer != nullptr)
	{
		_weights = new float[layerSize * previousLayer->_layerSize];
		_deltaWeights = new float[layerSize * previousLayer->_layerSize];
		_biases = new float[layerSize];
		_deltaBiases = new float[layerSize];
		_error = new float[layerSize];
		if (momentum > 0.0f)
		{
			_weightMomentum = new float[layerSize * previousLayer->_layerSize];
			_biasMomentum = new float[layerSize];
		}
		for (int i = 0; i < layerSize; ++i)
		{
			_biases[i] = 2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f;
			_deltaBiases[i] = 0.0f;
			if (momentum > 0.0f)
			{
				_biasMomentum[i] = 0.0f;
			}
			_error[i] = 0.0f;
			for (int j = 0; j < previousLayer->_layerSize; ++j)
			{
				_weights[i * _prevLayer->_layerSize + j] = 2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f;
				_deltaWeights[i * _prevLayer->_layerSize + j] = 0.0f;
				if (momentum > 0.0f)
				{
					_weightMomentum[i * _prevLayer->_layerSize + j] = 0.0f;
				}
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

void AnnUtilities::Layer::propagateForward()
{
	if (_actfunc == AnnUtilities::ACTFUNC::SIGMOID)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_outputs[i] = 0.0f;
			for (int j = 0; j < _prevLayer->_layerSize; ++j)
			{
				_outputs[i] += _weights[i * _prevLayer->_layerSize + j] * _prevLayer->_outputs[j];
			}
			_outputs[i] += _biases[i];
			_outputs[i] = 1.0f / (1.0f + expf(-_outputs[i]));
		}
	}
	else if (_actfunc == AnnUtilities::ACTFUNC::RELU)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_outputs[i] = 0.0f;
			for (int j = 0; j < _prevLayer->_layerSize; ++j)
			{
				_outputs[i] += _weights[i * _prevLayer->_layerSize + j] * _prevLayer->_outputs[j];
			}
			_outputs[i] += _biases[i];
			if (_outputs[i] < 0)
			{
				_outputs[i] = 0;
			}
		}
	}
	else if (_actfunc == AnnUtilities::ACTFUNC::LEAKY_RELU)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_outputs[i] = 0.0f;
			for (int j = 0; j < _prevLayer->_layerSize; ++j)
			{
				_outputs[i] += _weights[i * _prevLayer->_layerSize + j] * _prevLayer->_outputs[j];
			}
			_outputs[i] += _biases[i];
			if (_outputs[i] < 0)
			{
				_outputs[i] = 0.01f * _outputs[i];
			}
		}
	}
	else if (_actfunc == AnnUtilities::ACTFUNC::TANH)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_outputs[i] = 0.0f;
			for (int j = 0; j < _prevLayer->_layerSize; ++j)
			{
				_outputs[i] += _weights[i * _prevLayer->_layerSize + j] * _prevLayer->_outputs[j];
			}
			_outputs[i] += _biases[i];
			_outputs[i] = tanhf(_outputs[i]);
		}
	}
}

void AnnUtilities::Layer::propagateBackward()
{
	calculateError();
	calculateDerivative();
	calculateDelta();
}

void AnnUtilities::Layer::propagateBackward(const float* const label)
{
	for (int i = 0; i < _layerSize; ++i)
	{
		_error[i] = label[i] - _outputs[i];
	}
	calculateDerivative();
	calculateDelta();
}

void AnnUtilities::Layer::calculateError()
{
	for (int i = 0; i < _layerSize; ++i)
	{
		_error[i] = 0.0f;
		for (int j = 0; j < _nextLayer->_layerSize; ++j)
		{
			_error[i] += _nextLayer->_error[j] * _nextLayer->_weights[j * _layerSize + i];
		}
	}
}

void AnnUtilities::Layer::calculateDerivative()
{
	if (_actfunc == AnnUtilities::ACTFUNC::SIGMOID)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_error[i] = _error[i] * AnnUtilities::dSigmoid(_outputs[i]);
		}
	}
	else if (_actfunc == AnnUtilities::ACTFUNC::RELU)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_error[i] = _error[i] * AnnUtilities::dRelu(_outputs[i]);
		}
	}
	else if (_actfunc == AnnUtilities::ACTFUNC::LEAKY_RELU)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_error[i] = _error[i] * AnnUtilities::dLeakyRelu(_outputs[i]);
		}
	}
	else if (_actfunc == AnnUtilities::ACTFUNC::TANH)
	{
		for (int i = 0; i < _layerSize; ++i)
		{
			_error[i] = _error[i] * AnnUtilities::dTanh(_outputs[i]);
		}
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
	unsigned int node;
	for (int i = 0; i < _layerSize; ++i)
	{
		if (_momentum > 0.0f)
		{
			_biases[i] += learningRate * _deltaBiases[i] / epochs + _biasMomentum[i];
			_biasMomentum[i] = _momentum * _biasMomentum[i] + _momentum * _deltaBiases[i];
		}
		else
		{
			_biases[i] += learningRate * _deltaBiases[i] / epochs;
		}
		_deltaBiases[i] = 0.0f;
		for (int j = 0; j < _prevLayer->_layerSize; ++j)
		{
			node = i * _prevLayer->_layerSize + j;
			if (_momentum > 0.0f)
			{
				_weights[node] += learningRate * _deltaWeights[node] / epochs + _weightMomentum[node];
				_weightMomentum[node] = _momentum * _weightMomentum[node] + _momentum * _deltaWeights[node];
			}
			else
			{
				_weights[node] += learningRate * _deltaWeights[node] / epochs;
			}
			_deltaWeights[node] = 0.0f;
		}
	}
}