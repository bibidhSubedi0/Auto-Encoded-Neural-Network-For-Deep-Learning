#include"Neuron.h"
#include<math.h>
using namespace std;

Neuron::Neuron(double val)
{
    this->val = val;
    //Activate();
    //Derive();
}

void Neuron::setVal(double val)
{
    this->val = val;
    //Activate();
    //Derive();
}


void Neuron::Activate()
{
    // Using the relu
    // this->activatedVal = this->val > 0 ? this->val : 0;

    // Leaky Relu perfoms extermly well
    // If value is positive, use it, otherwise multiply by a small constant alpha (e.g., 0.01)
    this->activatedVal = this->val > 0 ? this->val : 0.01 * this->val;
}

void Neuron::Derive()
{
    // Derivative for relu
    // this->derivedVal = this->val > 0 ? 0 : 1;
    
    
    // 1 if value > 0, else alpha (e.g., 0.01) for negative values
    this->derivedVal = this->val > 0 ? 1 : 0.01;  // Assuming alpha = 0.01
}
 
void Neuron::ActivateFinal()
{
    // Sigmoid activation function: 1 / (1 + exp(-x))
    this->activatedVal = 1.0 / (1.0 + exp(-1 * this->val));
    
}

void Neuron::DeriveFinal()
{
    // Derivative of Sigmoid: sigma(x) * (1 - sigma(x))
    this->derivedVal = this->activatedVal * (1.0 - this->activatedVal);

}

//void Neuron::ActivateSoftmax(std::vector<Neuron*> neurons)
//{
//    // Compute exponentials and their sum
//    double maxVal = -INFINITY;
//    for (const auto& neuron : neurons)
//        maxVal = maxVal > neuron->getVal() ? maxVal : neuron->getVal();; // Avoid overflow by normalizing with max
//
//    vector<double> expVals(neurons.size());
//    for (size_t i = 0; i < neurons.size(); i++)
//        expVals[i] = exp(neurons[i].val - maxVal);
//
//    double sumExp = accumulate(expVals.begin(), expVals.end(), 0.0);
//
//    // Normalize to get softmax values
//    for (size_t i = 0; i < neurons.size(); i++)
//        neurons[i].activatedVal = expVals[i] / sumExp;
//}

void Neuron::setActivatedVal(double val)
{
    this->activatedVal = val;
}


