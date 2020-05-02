#pragma once
#include "Functions.h"

namespace AnnUtilities
{
	struct ANNSettings
	{
		/// Activation and derivative function used for hidden layers.
		AnnUtilities::ACTFUNC _hiddenActivationFunction = AnnUtilities::ACTFUNC::TANH;

		/// Activation and derivative function used for output layer.
		AnnUtilities::ACTFUNC _outputActivationFunction = AnnUtilities::ACTFUNC::SIGMOID;

		/// Size of the network's input.
		int _inputSize = 1;

		/// Size of the network's output
		int _outputSize = 1;

		/// Size of hidden layers
		int _hiddenSize = 1;

		/// Number of hidden layers between input and output layer
		int _numberOfHiddenLayers = 1;

		/// Learning rate multiplier
		float _learningRate = 0.1f;

		/// Momentum between 0.0f and 1.0f. If 0.0f, then no memory is allocated for momentum array.
		/// Momentum carries part of the previous delta values over when calculating new weights and biases.
		float _momentum = 0.0f;
	};
}
