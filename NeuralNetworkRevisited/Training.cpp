#include "Training.h"
#include <fstream>

Training::Training(vector<vector<double>> inputs, vector<vector<double>> outputs, vector<double> learning_rates, vector<vector<int>>topologies, int epoch) :inputs(inputs), outputs(outputs), learning_Rates(learning_rates), topologies(topologies),epoch(epoch){}

Network* Training::fit_Network(vector<double> input, vector<double> output, double learning_rate, vector<int> topology) {
	
	Network* Net = new Network(topology,learning_rate);
	Net->setCurrentInput(input);
	Net->setTarget(output);
	return Net;
}


void Training::train_Network()
{
	
		for (int learning_Rates_idx = 0; learning_Rates_idx < this->learning_Rates.size(); learning_Rates_idx++)
		{
			for (int topologies_indx = 0; topologies_indx < this->topologies.size(); topologies_indx++)
			{



				// Training the network for this particular configuration
				cout << "Traing for : " << endl;
				cout << "Learning Rate : " << learning_Rates[learning_Rates_idx] << endl;
				cout << "Topology : ";
				for (int lx : this->topologies[topologies_indx]) { cout << lx << " "; }
				cout << endl;





				Network* Net = new Network(this->topologies[topologies_indx], this->learning_Rates[learning_Rates_idx]);
				vector<double> avg_error_over_epoch;
				double error_for_epoch = 0;
				for (int epochs = 0; epochs < this->epoch; epochs++) 
				{
					cout << "Epoch : " << epochs << endl;;
					error_for_epoch = 0;
					for (int j = 0; j < this->inputs.size(); j++)
					{
						// Net = fit_Network(this->inputs[j], this->outputs[j], this->learning_Rates[learning_Rates_idx], this->topologies[topologies_indx]);

						Net->setCurrentInput(this->inputs[j]);
						Net->setTarget(this->outputs[j]);

						Net->forwardPropogation();
						Net->setErrors();
						error_for_epoch += Net->getGlobalError();

						vector<Matrix* > grad = Net->gardientComputation();
						vector<vector<Matrix*>> allGradientsInOneEpoch;
						allGradientsInOneEpoch.push_back(grad);
						vector<Matrix*> avgG = Net->averageGradients(allGradientsInOneEpoch);

						Net->updateWeights();
					}
					avg_error_over_epoch.push_back(error_for_epoch / 7);

				}

				// Checking for best network
				// This code does not work :( SED
				if (error_for_epoch<best.error){
					best.topology = topologies[topologies_indx];
					best.lr = learning_Rates[learning_Rates_idx];
					best.Net = Net;
				}

				std::ofstream outFile("error_over_epoch.csv");
				if (outFile.is_open()) {
					for (double error : avg_error_over_epoch) {
						outFile << error << "\n";
					}
					outFile.close();
				}






			}
		}
}


Network* Training::get_best_network()
{
	return best.Net;
}
double Training::get_best_lr()
{
	return best.lr;
}
vector<int> Training::get_best_topology()
{
	return best.topology;
}