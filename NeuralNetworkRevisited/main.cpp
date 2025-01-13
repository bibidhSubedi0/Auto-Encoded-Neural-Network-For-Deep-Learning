#include "Training.h"
#include <fstream>
using namespace std;


int main()
{
	vector<vector<double>> inputs{
		{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, { 0, 1, 1}, { 1, 0, 0}, { 1, 0, 1}, { 1, 1, 0}, { 1, 1, 1} 

	};

	vector<vector<double>> targets{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{ 0, 1, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 1, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 1, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 1, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 1 },

	};


	/*vector<vector<double>> test{
		{1, 0, 1, 0}, 
		{1, 0, 1, 1}, 
		{1, 1, 0, 0}, 
		{1, 1, 0, 1}, 
		{1, 1, 1, 0}, 
		{1, 1, 1, 1}
	};*/

	vector<double> learning_rates = { 0.01 };// { 0.01, 0.1, , 1 };
	vector<vector<int>> topologies = { { (int)inputs[0].size(),8,(int)targets[0].size()} }; // { {4, 8, 4}, { 4,8,16,8,4 },  };

	Training* t = new Training(inputs, targets, learning_rates, topologies, 100);
	t->train_Network();


	Network* best_network = t->get_best_network();
	//double best_lr = t->get_best_lr();
	//vector<int> best_topology = t->get_best_topology();

	//cout << "Best Learning Rate : " << best_lr << endl;
	//cout << "Best Topology : ";
	//for (auto tp : best_topology) { cout << tp << " ";}
	//cout << endl;

	

	for (int j = 0; j < inputs.size(); j++)
	{
		best_network->setCurrentInput(inputs[j]);
		best_network->setTarget(inputs[j]);
		best_network->forwardPropogation();
		best_network->printToConsole();
		cout << endl;
		cout << "-------------------------------------------------" << endl;
	}
	// Final weigts and biases
	best_network->printWeightMatrices();

	best_network->printBiases();
}
