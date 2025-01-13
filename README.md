
# Neural Network Framework for Deep Learning

This repository provides a comprehensive framework for building and training neural networks, facilitating the development of deep learning models from scratch. It offers a modular and extensible architecture, enabling users to experiment with various neural network architectures and training algorithms.

## Features

- **Modular Design**: The framework is structured into distinct modules, each responsible for specific functionalities such as data preprocessing, model construction, and training. This design promotes code reusability and simplifies maintenance.

- **Extensibility**: Users can easily extend the framework by adding new layers, activation functions, or optimization algorithms, allowing for experimentation with different model architectures and training strategies.

- **Comprehensive Documentation**: Detailed documentation is provided to guide users through the setup, usage, and customization of the framework.

-> We all know this is chat gpt generated, but below is something useful

## Installation

To install the framework, clone the repository and install the required dependencies:


git clone https://github.com/bibidhSubedi0/Neural-Network-Framework-For-Deep-Learning.git
cd Neural-Network-Framework-For-Deep-Learning


## Usage

After installation, you can build and train a neural network as follows:

```c++

# Set the netwok parameters
vector<vector<double>> inputs{
	{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, { 0, 1, 1}, { 1, 0, 0}, { 1, 0, 1}, { 1, 1, 0}, { 1, 1, 1} };

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

vector<double> learning_rates = { 0.01,0.1,0.001,0.5}
vector<vector<int>> topologies = { { inputs[0].size(),8,targets[0].size(),inputs[0].size(),16,8,16,targets[0].size()} }; 

# Train the network
network.train(training_data, training_labels, epochs=10, batch_size=32)

# Get the best network
Network* best_network = t->get_best_network();

# Make a prediction
for (int j = 0; j < inputs.size(); j++)
{
	best_network->setCurrentInput({0,0,1);
	best_network->setTarget({0,1,0,0,0,0,0,0);
	best_network->forwardPropogation();
	best_network->printToConsole();
}

```

## Contributing

I am very very open to contributions as there are worlds of feature that can be implemented. Also i am aware that this may not be the most optimally structured, written or organized project, but hey I am still learning stuff.

1. Fork the repository.
2. Create a new branch for your feature or bug fix
5. Submit a pull request with a clear description of your changes.

