#pragma once
#include "Functions.h"

namespace AnnUtilities
{
	class Layer
	{
	private:
		AnnUtilities::ACTFUNC _actfunc;
		float* _deltaWeights = nullptr;
		float* _deltaBiases = nullptr;
		float* _weightMomentum = nullptr;
		float* _biasMomentum = nullptr;
		float* _error = nullptr;

		/// Momentum value. If momentum is 0.0f, no memory is allocated for momentum.
		float _momentum = 0.0f;
		void calculateError();
		void calculateDerivative();
		void calculateDelta();

	public:
		int _layerSize;
		Layer* _prevLayer = nullptr;
		Layer* _nextLayer = nullptr;
		float* _outputs = nullptr;
		float* _biases = nullptr;
		float* _weights = nullptr;
		Layer(Layer* previousLayer, int layerSize, float momentum, AnnUtilities::ACTFUNC actfunc);
		~Layer();

		void propagateForward();

		/// Propagate backward. Delta weights and biases are cumulatively added on each back propagation.
		void propagateBackward();

		/// Propagate backward with error of (target - output).
		void propagateBackward(const float* const label);

		/// Update weights and biases. Update will zero delta values. If the layer has momentum, update will calculate new momentum values.
		void update(const float learningRate, const int epochs);

		void setNextLayer(Layer* nextLayer) { _nextLayer = nextLayer; }

		/// Sets the layer's outputs. Used on input layer to set the networks input.
		void setOutputs(const float* const outputs) { for (int i = 0; i < _layerSize; ++i) { _outputs[i] = outputs[i]; }; }

		/// Return the output array.
		float* getOutput() const { return _outputs; }

		/// Return value of a single node.
		float getOutput(const int node) const { return _outputs[node]; }
	};
}