#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include<iostream>
#include"Neuron.h"
#include<vector>
#include"Matrix.h"
using namespace std;

class Layer
{
public:
    
    // Constructor for 
    Layer(int size);
    Layer(int size,bool last);
    void setVal(int index, double val);
    void setActivatedVal(int index, double val);

    Matrix* convertTOMatrixVal();
    Matrix* convertTOMatrixActivatedVal();
    Matrix* convertTOMatrixDerivedVal();
    Layer* feedForward(Matrix* LastWeights, Matrix* LastBias, bool isFirst,bool isLast);
    vector<Neuron*> getNeurons();
    Neuron* getNeuron(int);
    int getSize();


private:
    int size;
    vector<Neuron*> neurons;
};


#endif