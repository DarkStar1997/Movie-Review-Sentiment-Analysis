#include "NN.hpp"
int main()
{
	Net net({25286, 100, 50, 1});
	net.setActivationFunction(1, act::sigmoid, act::sigmoidDerivative);
	net.setActivationFunction(2, act::sigmoid, act::sigmoidDerivative);
	net.loadModel("model");
	while(true)
	{
		std::cout<<"Edit test.txt, enter n to quit\n";
		char str; std::cin>>str;
		if(str == 'n' || str == 'N')
			break;
		system("python3 get_file_tfidf.py > test1.txt");
		net.setInputFile("test1.txt");
		net.feedForward();
		auto ans = net.getResults();
		for(const auto& i : ans)
			std::cout << i <<' ';
		std::cout<<'\n';
	}
}
