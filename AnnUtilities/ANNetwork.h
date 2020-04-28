#pragma once
#include "Functions.h"


namespace AnnUtilities
{
	class Layer;
	struct InputData;

	class ANNetwork
	{
	private:

	public:
		Layer* _inputLayer = nullptr;
		Layer* _outputLayer = nullptr;
		ANNetwork();
		~ANNetwork();
		void propagateForward();
		void propagateBackward(const float* const labels);
		void Init(const int inputSize, const int hiddenSize, const int outputSize, const int hiddenLayers, AnnUtilities::ACTFUNC actfuncHidden, AnnUtilities::ACTFUNC actfuncOutput);
		void Epoch(const InputData* const inputData, const int inputSize, const float learningRate);
		void update(const int batchSize, const float learningRate);
		float* Test(const float* const inputData);
		void Clean();
	};
}