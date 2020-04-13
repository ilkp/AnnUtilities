#pragma once



namespace AnnUtilities
{
	class Layer;
	struct InputData;

	class Network
	{
	private:
		Layer* _inputLayer = nullptr;
		Layer* _outputLayer = nullptr;
		void propagateForward();
		void propagateBackward(const float* const labels);

	public:
		Network();
		~Network();
		void Init(const int inputSize, const int hiddenSize, const int outputSize, const int hiddenLayers,
			float(*activationFuncHiddenL)(float), float(*activationFuncOutputL)(float), float(*derivativeFuncHiddenL)(float), float(*derivativeFuncOutputL)(float));
		void Epoch(const InputData* const inputData, const int inputSize, const float learningRate);
		float* Test(const float* const inputData);
		void Clean();
	};
}