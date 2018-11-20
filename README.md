# Movie Review Sentiment Analysis from Scratch

In this repository, we have implemented a movie review sentiment analysis tool which rates the review and returns a fractional value between 0 and 1, where, 0 indicates negative and 1 being positive. For preprocessing, we have used the TFIDF algorithm from the Scikit-learn package. The main training has been done on an Artificial Neural Network (ANN) written in C++ and [Arrayfire](http://arrayfire.org/docs/index.htm) which is capable of accelerating Linear Algebra operations on CPU as well as GPU using Intel MKL, OpenCL and CUDA.

A simple re-implementable version of the Neural Network written in C++/ArrayFire can be found [here](https://github.com/codebuddha/Neural_Networks_from_Scratch).

## Note:

The integration of the Python code for TFIDF and the C++ code for the ANN has been done very naively using a system call: `system("python3 get_file_tfidf.py > test1.txt");` in [test.cpp](https://github.com/DarkStar1997/Movie-Review-Sentiment-Analysis/blob/master/test.cpp) avoiding the usage of any wrapper code. 
