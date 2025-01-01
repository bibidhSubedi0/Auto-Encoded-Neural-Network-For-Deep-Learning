#include "TotalNetwork.h"
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <fstream>


Network::Network(vector<int> topology, double lr)
{
    this->learningRate = lr;
    this->topology = topology;
    this->topologySize = topology.size();

    for (int i = 0; i < topologySize; i++)
    {

        // If it is last layer treat it seperately
        if (i == topologySize - 1)
        {
            Layer* l = new Layer(topology[i],1);
            this->layers.push_back(l);
        }

        // If it is not last layer
        else{
        Layer* l = new Layer(topology[i]);
        this->layers.push_back(l);
        }
    }


    for (int i = 0; i < topologySize - 1; i++)
    {
        Matrix* mw = new Matrix(topology[i], topology[i + 1], true);
        this->weightMatrices.push_back(mw);

        Matrix* mb = new Matrix(1, topology[i + 1], false);
        this->BaisMatrices.push_back(mb);
    }

    histErrors.push_back(1);
}


double Network::getGlobalError()
{
    return this->error;
}

double Network::lastEpoachError()
{
    return histErrors[histErrors.size() - 1];
}

void Network::printHistErrors()
{
    cout << "\n Printing Errors from all epochs:" << endl;
    for (int i = 0; i < this->histErrors.size(); i++)
    {
        cout << "Epoch: " << i << ", Error:" << histErrors.at(i) << " \n ";
    }
}

void Network::saveHistErrors()
{
    ofstream outFile("error_vs_epoch.csv");

    if (outFile.is_open())
    {

        outFile << "Epoch,Error\n";

        for (size_t i = 0; i < this->histErrors.size(); ++i)
        {
            outFile << i << "," << histErrors[i] << "\n";
        }
        outFile.close();
    }
    else
    {
        cerr << "Unable to open file for writing.\n";
    }
}

vector<double> Network::gethisterrors()
{
    return histErrors;
}

double Network::getLearningRate()
{
    return learningRate;
}



void Network::setTarget(vector<double> target)
{
    this->target = target;
}

void Network::printErrors()
{
    // cout<<"This Iteration Error"<<endl;
    // for(auto err : this->errors)
    // {
    //     cout<<"Ex : "<<err<<"  ";
    // }
    // cout<<endl;

    // cout<<"Historical Errors"<<endl;
    // for(auto err : this->errors)
    // {
    //     cout<<"Eh : "<<err<<"  ";
    // }
    // cout<<endl;

    cout << "Total Error : " << this->error << endl;
}

void Network::setCurrentInput(vector<double> input)
{
    this->input = input;

    for (int i = 0; i < input.size(); i++)
    {
        this->layers[0]->setVal(i, input[i]);
    }
}

void Network::printToConsole()
{
    // Print the inputs to the network
    for (int i = 0; i < input.size(); i++)
    {
        cout << input.at(i) << "\t\t";
    }
    cout << endl;

    // Print the outputs to the network
    for (int i = 0; i < layers.at(layers.size() - 1)->getSize(); i++)
    {
        cout << layers.at(layers.size() - 1)->getNeurons().at(i)->getActivatedVal() << "\t";
    }
}

Layer* Network::GetLayer(int nth)
{
    return layers[nth];
}

void Network::printWeightMatrices()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Weights for Hidden Layer : " << i + 1 << endl;
        weightMatrices[i]->printToConsole();
    }
}

void Network::forwardPropogation()
{
    for (int i = 0; i < layers.size() - 1; i++)
    {
        layers[i + 1] = layers[i]->feedForward(weightMatrices[i], BaisMatrices[i], (i == 0),(i==layers.size()-2));
    }
}

void Network::printBiases()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Bias for Hidden Layer : " << i + 1 << endl;
        BaisMatrices[i]->printToConsole();
    }
}



void Network::setErrors()
{
    if (this->target.size() == 0)
    {
        cerr << "No target found" << endl;
        assert(false);
    }

    if (target.size() != layers[layers.size() - 1]->getNeurons().size())
    {
        cerr << "The size of the target is not equal to the size of the output" << endl;
        assert(false);
    }

    errors.resize(target.size());

    this->error = 0;
    int outputLayerIndx = this->layers.size() - 1;
    vector<Neuron*> outputNeurons = this->layers[outputLayerIndx]->getNeurons();
    errorDerivatives.resize(errors.size());

    // -------------------------MSE---------------------------------------------
    //for (int i = 0; i < target.size(); i++)
    //{
    //    double req = target[i];
    //    double act = outputNeurons[i]->getActivatedVal();
    //    this->errors[i] = 0.5 * pow(abs((req - act)), 2);
    //    errorDerivatives[i] = act - req;
    //    this->error += errors[i];
    //}
   
    
    
    //------------------------Cross Entropy----------------------------------------
    for (int i = 0; i < target.size(); i++)
    {
        double act = target[i];
        double pred = outputNeurons[i]->getActivatedVal();
        this->errors[i] = act * std::log(pred) + (1 - act) * std::log(1 - pred);

        // These are baseically the gadeints
        errorDerivatives[i] = -((act / pred) - ((1 - act) / (1 - pred))); 

        // This is the total cross entropy error
        this->error += this->errors[i]; //
    }


}



void Network::saveThisError(double error)
{
    this->histErrors.push_back(error);
}

vector<Matrix*> Network::gardientComputation()
{

    Matrix* gradients;
    Matrix* DerivedValuesFromOtoH;
    Matrix* lastGradient;
    Matrix* tranposedWeightMatrices;
    Matrix* hiddenDerived;
    int outputLayerIndex = this->topology.size() - 1;
    vector<Matrix*> allGradients;

    gradients = new Matrix(
        1,
        this->topology.at(outputLayerIndex),
        false);
    DerivedValuesFromOtoH = this->layers.at(outputLayerIndex)->convertTOMatrixDerivedVal();
    for (int i = 0; i < this->topology.at(outputLayerIndex); i++)
    {
        double e = this->errorDerivatives.at(i);
        double y = DerivedValuesFromOtoH->getVal(0, i);
        double g = e * y;
        gradients->setVal(0, i, g);
    }
    Matrix* temp = new Matrix(1, this->topology.at(outputLayerIndex), false);
    allGradients.push_back(*temp + gradients);

    delete DerivedValuesFromOtoH;
    for (int i = (outputLayerIndex - 1); i > 0; i--)
    {
        Matrix* t = new Matrix(1, topology.at(i), false);
        lastGradient = new Matrix(*gradients);
        delete gradients;

        tranposedWeightMatrices = this->weightMatrices.at(i)->tranpose();
        gradients = new Matrix(
            1,
            this->topology.at(i),
            false);


        gradients = *lastGradient * tranposedWeightMatrices;

        hiddenDerived = this->layers.at(i)->convertTOMatrixDerivedVal();

        for (int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++)
        {
            double g = gradients->getVal(0, colCounter) * hiddenDerived->getVal(0, colCounter);
            gradients->setVal(0, colCounter, g);
        }

        allGradients.push_back(*t + gradients);
        delete lastGradient;
        delete tranposedWeightMatrices;
        delete hiddenDerived;
    }

    return allGradients;
}





void Network::updateWeights()
{
    vector<Matrix*> newWeights;
    Matrix* deltaWeights;
    Matrix* gradients;

    Matrix* gradientsTransposed;
    Matrix* PreviousLayerActivatedVals;
    Matrix* tempNewWeights;
    Matrix* lastGradient;
    Matrix* tranposedWeightMatrices;
    Matrix* hiddenDerived;
    Matrix* transposedHidden;
    int outputLayerIndex = topology.size() - 1;

    gradients = GradientMatrices[0];
    gradientsTransposed = gradients->tranpose();
    PreviousLayerActivatedVals = this->layers.at(outputLayerIndex - 1)->convertTOMatrixActivatedVal();
    deltaWeights = new Matrix(
        gradientsTransposed->getNumRow(),
        PreviousLayerActivatedVals->getNumCols(),
        false);

    deltaWeights = *gradientsTransposed * PreviousLayerActivatedVals;

    tempNewWeights = new Matrix(
        this->topology.at(outputLayerIndex - 1),
        this->topology.at(outputLayerIndex),
        false);


    for (int r = 0; r < this->topology.at(outputLayerIndex - 1); r++)
    {
        for (int c = 0; c < this->topology.at(outputLayerIndex); c++)
        {

            double originalValue = this->weightMatrices.at(outputLayerIndex - 1)->getVal(r, c);
            double deltaValue = deltaWeights->getVal(c, r);
            deltaValue = this->learningRate * deltaValue;

            tempNewWeights->setVal(r, c, (originalValue - deltaValue));
        }
    }

    newWeights.push_back(new Matrix(*tempNewWeights));
    delete gradientsTransposed;
    delete PreviousLayerActivatedVals;
    delete tempNewWeights;
    delete deltaWeights;

    int gmctr = 1;
    for (int i = (outputLayerIndex - 1); i > 0; i--)
    {
        lastGradient = new Matrix(*gradients);
        delete gradients;

        tranposedWeightMatrices = this->weightMatrices.at(i)->tranpose();

        gradients = new Matrix(
            lastGradient->getNumRow(),
            tranposedWeightMatrices->getNumCols(),
            false);

        gradients = *lastGradient * tranposedWeightMatrices;

        hiddenDerived = this->layers.at(i)->convertTOMatrixDerivedVal();

        for (int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++)
        {
            double g = gradients->getVal(0, colCounter) * hiddenDerived->getVal(0, colCounter);
            gradients->setVal(0, colCounter, g);
        }
        gradients = GradientMatrices[gmctr];
        gmctr++;

        if (i == 1)
        {
            PreviousLayerActivatedVals = this->layers.at(0)->convertTOMatrixVal();
        }
        else
        {
            PreviousLayerActivatedVals = this->layers.at(i - 1)->convertTOMatrixActivatedVal();
        }

        transposedHidden = PreviousLayerActivatedVals->tranpose();

        deltaWeights = new Matrix(
            transposedHidden->getNumRow(),
            gradients->getNumCols(),
            false);

        deltaWeights = *transposedHidden * gradients;

        tempNewWeights = new Matrix(
            this->weightMatrices.at(i - 1)->getNumRow(),
            this->weightMatrices.at(i - 1)->getNumCols(),
            false);

        for (int r = 0; r < tempNewWeights->getNumRow(); r++)
        {
            for (int c = 0; c < tempNewWeights->getNumCols(); c++)
            {
                double originalValue = this->weightMatrices.at(i - 1)->getVal(r, c);
                double deltaValue = deltaWeights->getVal(r, c);

                deltaValue = this->learningRate * deltaValue;

                tempNewWeights->setVal(r, c, (originalValue - deltaValue));
            }
        }

        newWeights.push_back(new Matrix(*tempNewWeights));

        delete lastGradient;
        delete tranposedWeightMatrices;
        delete hiddenDerived;
        delete PreviousLayerActivatedVals;
        delete transposedHidden;
        delete tempNewWeights;
        delete deltaWeights;
    }
    delete gradients;

    for (int i = 0; i < this->weightMatrices.size(); i++)
    {
        delete this->weightMatrices[i];
    }

    this->weightMatrices.clear();
    reverse(newWeights.begin(), newWeights.end());
    weightMatrices = newWeights;
}





vector<Matrix*> Network::averageGradients(vector<vector<Matrix*>> accgrad)
{
    size_t numColumns = accgrad[0].size();
    vector<Matrix*> averageMatrices(numColumns, nullptr);


    for (size_t col = 0; col < numColumns; ++col)
    {
        averageMatrices[col] = new Matrix(*accgrad[0][col]);

        for (int i = 0; i < averageMatrices[col]->getNumRow(); ++i)
        {
            for (int j = 0; j < averageMatrices[col]->getNumCols(); ++j)
            {
                averageMatrices[col]->setVal(i, j, 0.0);
            }
        }
    }




    for (int avgmahunuparne = 0; avgmahunuparne < numColumns; avgmahunuparne++)
    {
        for (int i = 0; i < accgrad.size(); i++)
        {
            averageMatrices[avgmahunuparne] = *averageMatrices[avgmahunuparne] + accgrad[i][avgmahunuparne];
        }
    }

    size_t numInputs = accgrad.size();
    for (size_t col = 0; col < numColumns; ++col)
    {
        averageMatrices[col]->divideByScalar(numInputs);
    }
    this->GradientMatrices.resize(averageMatrices.size());
    this->GradientMatrices = averageMatrices;
    return averageMatrices;
}
