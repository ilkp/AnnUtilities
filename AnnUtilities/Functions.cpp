#include "Functions.h"
#include <math.h>


float AnnUtilities::sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float AnnUtilities::dSigmoid(float x)
{
	return x * (1.0f - x);
}

float AnnUtilities::relu(float x)
{
	if (x < 0.0f)
	{
		return 0.0f;
	}
	return x;
}
float AnnUtilities::dRelu(float x)
{
	if (x < 0)
	{
		return 0.0f;
	}
	return 1.0f;
}

float AnnUtilities::leakyRelu(float x)
{
	if (x < 0)
	{
		return 0.01f * x;
	}
	return x;
}

float AnnUtilities::dLeakyRelu(float x)
{
	if (x < 0)
	{
		return 0.01f;
	}
	return 1.0f;
}

float AnnUtilities::hypTanh(float x)
{
	return tanhf(x);
}

float AnnUtilities::dTanh(float x)
{
	return 1.0f - x * x;
}