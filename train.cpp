#include "NN.hpp"
int main()
{
	Net net({25286, 100, 50, 1});
	//Net net({25286, 100, 100, 50, 1});

	net.setActivationFunction(1, act::sigmoid, act::sigmoidDerivative);
	net.setActivationFunction(2, act::sigmoid, act::sigmoidDerivative);

//	net.setActivationFunction(1, act::sigmoid, act::sigmoidDerivative);
//	net.setActivationFunction(2, act::sigmoid, act::sigmoidDerivative);
//	net.setActivationFunction(3, act::sigmoid, act::sigmoidDerivative);
//	net.setActivationFunction(4, act::sigmoid, act::sigmoidDerivative);

	net.setInputFile("/home/rohan/eclipse-workspace/ArrayFire_Tests/X_tftdf_1980_25286_ALL.csv");
	net.setOutputFile("/home/rohan/eclipse-workspace/ArrayFire_Tests/Y_1980_ALL.csv");
	net.setLearningRate(0.001);
	for(int i = 1; i <= 10000; i++)
	{
		//net.shuffle();
		net.feedForward();
		net.backPropagate();
		af::array error = net.A[net.layout.size() - 1] - net.Y;
		af::array mean = af::mean(af::abs(error));
		af::sync(0);
		af_print(mean);
		af::sync(0);
		std::cout << "epoch_terminated: " << i << std::endl;
	}
	net.save("model");
}
