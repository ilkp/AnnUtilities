#pragma once
#include "Functions.h"

namespace AnnUtilities
{
	struct ANNSettings
	{
		AnnUtilities::ACTFUNC _hiddenActicationFunction = AnnUtilities::ACTFUNC::TANH;
		AnnUtilities::ACTFUNC _outputActicationFunction = AnnUtilities::ACTFUNC::SIGMOID;
		int _inputSize = 1;
		int _outputSize = 1;
		int _hiddenSize = 1;
		int _numberOfHiddenLayers = 1;
		float _learningRate = 0.1f;
		float _momentum = 0.0f;
		float _maxMomentum = 0.0f;
	};
}
