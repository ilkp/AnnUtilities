#include "Functions.h"
#include <math.h>


/// Sigmoid function. Outputs value between 0.0f and 1.0f. Slow to calculate so recommended to use for only the output layer.
float AnnUtilities::sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

/// Derivative of Sigmoid function
float AnnUtilities::dSigmoid(float x)
{
	return x * (1.0f - x);
}

/// Rectified linear unit function. Outputs 0 when x < 0, and outputs x when x >= 0.
float AnnUtilities::relu(float x)
{
	if (x < 0.0f)
	{
		return 0.0f;
	}
	return x;
}

/// Derivative of ReLu
float AnnUtilities::dRelu(float x)
{
	if (x < 0)
	{
		return 0.0f;
	}
	return 1.0f;
}

/// Leaky rectified linear unit. Outputs 0.01 * x when x < 0, and outputs x when x >= 0.
float AnnUtilities::leakyRelu(float x)
{
	if (x < 0)
	{
		return 0.01f * x;
	}
	return x;
}

/// Derivative of linear rectified unit function
float AnnUtilities::dLeakyRelu(float x)
{
	if (x < 0)
	{
		return 0.01f;
	}
	return 1.0f;
}

/// Hyperbolic tangent function. Outputs tanhf(x) of math.h
float AnnUtilities::hypTanh(float x)
{
	return tanhf(x);
}

/// Derivative of hyperbolic tangent function.
float AnnUtilities::dTanh(float x)
{
	return 1.0f - x * x;
}