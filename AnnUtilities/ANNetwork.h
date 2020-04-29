#pragma once
#include "ANNSettings.h"


namespace AnnUtilities
{
	class Layer;
	struct InputData;

	class ANNetwork
	{
	private:

	public:
		AnnUtilities::ANNSettings _settings;
		Layer* _inputLayer = nullptr;
		Layer* _outputLayer = nullptr;
		ANNetwork();
		~ANNetwork();
		void propagateForward();
		void propagateBackward(const float* const labels);
		void init();
		void epoch(const InputData* const inputData, const int inputSize, const float learningRate);
		void update(const int batchSize);
	};
}