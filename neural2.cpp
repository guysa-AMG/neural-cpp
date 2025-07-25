#include <iostream>
#include <vector>
#include <cmath>
// Define the activation function (sigmoid)
double sigmoid(double x){
    return 1/(1+exp(-x));
}

//Define the loss function (mean squared error)
double mse_loss(const std::vector<double> & output , const std::vector<double> &target){
    double sum =0;
    for ( size_t i = 0; i < output.size();++i){
        sum += (output[i] - target[i])* (output[i] - target[i]);
    }
    return sum / output.size();
}

//Define the optimization algorithm (SGD)
void SGD(std::vector<double> &weights, const std::vector<double>&inputs, double learning_rate){
    for (size_t i = 0; i < weights.size();++i){
        weights[i] -= learning_rate * inputs[i];
    }
}

int main(){
    //initialize the network parameter
    std::vector<double> weights(3);
    weights[0] = 2.5;
    weights[1] = 3.7;

    //Define the input data
    std::vector<double> inputs(10,1.0);

    //Define the output data
    std::vector<double> targets(10,0.0);

    //Forward pass
    double outputs[10];
    for (size_t i =0;i<3;++i){
        outputs[i] = sigmoid(weights[i]*inputs[i]);
    }

    //Backward pass
    std::vector<double> error(10,0.0);
    for(size_t i=0;i<10;++i){
        double delta = targets[i] - outputs[i];
        error[i] += (outputs[i] * (1-outputs[i])) *delta;
    }

    //update the weights
    SGD(weights,inputs,0.01);

    //Print the results
    std::cout <<"output: ";
    for(size_t i =0 ;i<10;++i){
        std::cout<<outputs[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}