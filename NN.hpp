#include "Activations.hpp"
#include <iostream>
#include <random>
#include <algorithm>
class Net
{
//private:
public:     //SPOILER ALERT!!!!!!!!!!!!!!!!!!!!!!!!!
	std::vector<af::array> A;
	std::vector<af::array> dA;
	std::vector<af::array> W;
	std::vector<std::pair<af::array(*)(const af::array&),af::array(*)(const af::array&)>> activations;
	af::array Y;
	std::vector<int> layout;
	double alpha = 0.001;
	int iterations = 1;
public:
	Net(std::vector<int> tmp);
	void setInputMatrix(const std::vector<std::vector<float>> &x);
	void setInputFile(std::string file);
	void setOutputMatrix(const std::vector<std::vector<float>> &y);
	void setOutputFile(std::string file);
	void setActivationFunction(int layer, af::array(*acti)(const af::array&), af::array(*actiDerivative)(const af::array&));
	void setLearningRate(double x);
	void setIterations(int iter);
	void feedForward();
	void backPropagate();
	std::vector<float> getResults();
	void train();
	void predict(const std::vector<std::vector<float>> &x);
	void save(std::string pathname);
	void loadModel(std::string pathname);
	void shuffle();
	void shuffleInput(const std::vector<int> &indices);
	void shuffleOutput(const std::vector<int> &indices);
};

Net::Net(std::vector<int> tmp)
{
	for(int i = 0; i < tmp.size() - 1; i++)
		tmp[i]++;
	layout = tmp;
	A = std::vector<af::array>(layout.size());
	dA = std::vector<af::array>(layout.size());
	W = std::vector<af::array>(layout.size() - 1);
	af::setSeed(std::random_device()());
	for(int i = 0; i < W.size(); i++)
		W[i] = af::randn(layout[i], layout[i+1]);
	activations = std::vector<std::pair<af::array(*)(const af::array&),af::array(*)(const af::array&)>>(layout.size() - 1);
	for(int i = 0; i < activations.size() - 1; i++)
		activations[i] = std::make_pair(act::ReLu, act::ReLuDerivative);
	activations[activations.size() - 1] = std::make_pair(act::sigmoid, act::sigmoidDerivative);
}

void Net::setInputMatrix(const std::vector<std::vector<float>> &x)
{
	A[0] = af::array(x.size(), layout[0]);
	for(int i = 0; i < A[0].dims(0); i++)
		for(int j = 0; j < A[0].dims(1); j++)
			if(j == x[i].size())
				A[0](i, j) = 1.0;
			else
				A[0](i, j) = x[i][j];
}

void Net::setInputFile(std::string file)
{
	FILE *in=fopen(file.c_str(), "r");
	fseek(in, 0, SEEK_END);
	size_t size = ftell(in);
	std::vector<char> buffer(size);
	rewind(in);
	fread(buffer.data(), 1, size, in);
	fclose(in);
	std::vector<float> tmp; std::string tmp1 = "";
	for(const auto& i : buffer)
	{
		if( i == ' ' || i == '\n')
		{
			if(tmp1 != "")
				tmp.emplace_back((float)std::atof(tmp1.c_str()));
			tmp1 = "";
		}
		else
			tmp1 += i;
	}
	if(tmp1 != "")
		tmp.emplace_back((float)std::atof(tmp1.c_str()));
	tmp.shrink_to_fit();
	A[0] = af::array(layout[0] - 1, tmp.size() / (layout[0] - 1), tmp.data(), afHost);
	A[0] = A[0].T();
	A[0] = af::join(1, A[0], af::constant(1.0, A[0].dims(0)));
}

void Net::setOutputFile(std::string file)
{
	FILE *in=fopen(file.c_str(), "r");
	fseek(in, 0, SEEK_END);
	size_t size = ftell(in);
	std::vector<char> buffer(size);
	rewind(in);
	fread(buffer.data(), 1, size, in);
	fclose(in);
	std::vector<float> tmp; std::string tmp1 = "";
	for(const auto& i : buffer)
	{
		if( i == ' ' || i == '\n')
		{
			if(tmp1 != "")
				tmp.emplace_back((float)std::atof(tmp1.c_str()));
			tmp1 = "";
		}
		else
			tmp1 += i;
	}
	if(tmp1 != "")
		tmp.emplace_back((float)std::atof(tmp1.c_str()));
	tmp.shrink_to_fit();
	Y = af::array(layout[layout.size() - 1], tmp.size() / layout[layout.size() - 1], tmp.data(), afHost);
	Y = Y.T();
}

void Net::setOutputMatrix(const std::vector<std::vector<float>> &y)
{
	Y = af::array(y.size(), y[0].size());
	for(int i = 0; i < y.size(); i++)
		for(int j = 0; j < y[0].size(); j++)
			Y(i, j) = y[i][j];
}

void Net::setLearningRate(double x)
{
	alpha = x;
}

void Net::setIterations(int iter)
{
	iterations = iter;
}

void Net::feedForward()
{
	for(int i = 1; i < layout.size(); i++)
		A[i] = activations[i-1].first(af::matmul(A[i-1], W[i-1]));
}

void Net::backPropagate()
{
	for(int i = layout.size() - 1; i > 0; i--)
	{
		if(i == layout.size() - 1)
		{
			af::array error = A[i] - Y;
			dA[i] = error * activations[i-1].second(A[i]);
			W[i-1] = W[i-1] - alpha * af::matmul(A[i-1].T(), dA[i]);
		}
		else
		{
			dA[i] = af::matmul(dA[i+1], W[i].T()) * activations[i-1].second(A[i]);
			W[i-1] = W[i-1] - alpha * af::matmul(A[i-1].T(), dA[i]);
		}
	}
}

std::vector<float> Net::getResults()
{
	std::vector<float> res(A[layout.size() - 1].elements());
	A[layout.size() - 1].T().host(res.data());
	return res;
}

void Net::train()
{
	for(int iter = 1; iter <= iterations; iter++)
	{
		feedForward();
		backPropagate();
		if(iter % 1000 == 0)
		{
			af::array error = A[layout.size() - 1] - Y;
			af::array mean = af::mean(error);
			std::cout<<"After "<<iter<<" iterations, error =\n";
			af_print(mean);
		}
	}
}

void Net::predict(const std::vector<std::vector<float>> &x)
{
	setInputMatrix(x);
	feedForward();
}

void Net::setActivationFunction(int layer, af::array(*acti)(const af::array&), af::array(*actiDerivative)(const af::array&))
{
	activations[layer - 1] = std::make_pair(acti, actiDerivative);
}

void Net::save(std::string pathname)
{
	std::string tmp = "layer";
	for(int i = 0; i < W.size(); i++)
	{
		std::string key = tmp + std::to_string(i);
		if(i == 0)
			af::saveArray(key.c_str(), W[i], pathname.c_str());
		else
			af::saveArray(key.c_str(), W[i], pathname.c_str(), true);
	}
}

void Net::loadModel(std::string pathname)
{
	std::string tmp = "layer";
	for(int i = 0; i < W.size(); i++)
	{
		std::string key = tmp + std::to_string(i);
		W[i] = af::readArray(pathname.c_str(), key.c_str());
	}
}

void Net::shuffle()
{
	std::mt19937_64 rng;
	rng.seed(std::random_device()());
	std::vector<int> index(A[0].dims(0));
	for(int i = 0; i < index.size(); i++)
		index[i] = i;
	std::shuffle(index.begin(), index.end(), rng);
	shuffleInput(index);
	shuffleOutput(index);
}

void Net::shuffleInput(const std::vector<int> &indices)
{
	af::array tmp(A[0].dims(0), A[0].dims(1));
	for(int i = 0; i < indices.size(); i++)
		tmp(i, af::span) = A[0](indices[i], af::span);
	A[0] = tmp;
}

void Net::shuffleOutput(const std::vector<int> &indices)
{
	af::array tmp(Y.dims(0), Y.dims(1));
	for(int i = 0; i < indices.size(); i++)
		tmp(i, af::span) = Y(indices[i], af::span);
	Y = tmp;
}
