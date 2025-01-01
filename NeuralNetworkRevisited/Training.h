#pragma once
#include"TotalNetwork.h"

class Training
{
	vector<vector<double>> inputs;
	vector<vector<double>> outputs;
	vector<double> learning_Rates;
	vector<vector<int>> topologies;
	int epoch;
	struct best_network {
		double lr;
		vector<int> topology;
		Network* Net;
		double error;

		best_network(double lr = 0, vector<int> topology = {}, Network* Net = NULL, double error = INFINITY) :lr(lr), topology(topology), Net(Net), error(error) {}
	}best;
	
public:
	Training(vector<vector<double>>, vector<vector<double>>, vector<double>, vector<vector<int>>,int);
	Network* fit_Network(vector<double> input, vector<double> output,double learning_rate, vector<int> topology);
	void train_Network(); // The output is stored in the best_network structure
	
	Network* get_best_network();
	double get_best_lr();
	vector<int> get_best_topology();
};

