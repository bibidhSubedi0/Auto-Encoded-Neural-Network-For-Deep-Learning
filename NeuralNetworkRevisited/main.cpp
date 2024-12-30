#include "TotalNetwork.h"
using namespace std;


int main()
{
	vector<int> topology = { 3,4,3 };
	NN* Network = new NN(topology, 0.1);

	Network->setCurrentInput({ 1,0,1 });
	Network->setTarget({ 1,0,1 });



	for(int epochs =0;epochs<100;epochs++){
		Network->forwardPropogation();


		Network->setErrors();
		
		vector<Matrix* > grad = Network->gardientComputation();
		vector<vector<Matrix*>> allGradientsInOneEpoch;
		allGradientsInOneEpoch.push_back(grad);
		vector<Matrix*> avgG = Network->averageGradients(allGradientsInOneEpoch);
	
		Network->updateWeights();
	}

	Network->printToConsole();

}
