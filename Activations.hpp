/*
 * Sigmoid
 * ReLu
 * tanh
 */

#include <arrayfire.h>
namespace act
{
	af::array sigmoid(const af::array &x)
	{
		return 1.0 / (1.0 + af::exp(-x));
	}
	af::array sigmoidDerivative(const af::array &x)
	{
		return x * (1.0 - x);
	}
	af::array ReLu(const af::array &x)
	{
		af::array y = x;
		y(af::where(y < 0)) = 0;
		return y;
	}
	af::array ReLuDerivative(const af::array &x)
	{
		af::array y = x;
		y(af::where(y < 0)) = 0;
		y(af::where(y == 0)) = 0.01;
		y(af::where(y > 0)) = 1;
		return y;
	}
	af::array tanh(const af::array &x)
	{
		return af::tanh(x);
	}
	af::array tanhDerivative(const af::array &x)
	{
		af::array y = af::tanh(x);
		return 1.0 - (y*y);
	}
};
