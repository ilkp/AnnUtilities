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
		/// ANNetwork settings. Set this struct before calling init().
		AnnUtilities::ANNSettings _settings;
		Layer* _inputLayer = nullptr;
		Layer* _outputLayer = nullptr;
		ANNetwork();
		~ANNetwork();

		/// Propagate the network forward. After calling propagateForward(), output array of the last layer will contain the networks output.
		void propagateForward();

		/// Propagate backward. Error of the output layer will be (target - output).
		void propagateBackward(const float* const labels);

		/// Allocate and initialize Layers.
		void init();
		void epoch(const InputData* const inputData, const int inputSize, const float learningRate);
		void update(const int batchSize);
	};
}