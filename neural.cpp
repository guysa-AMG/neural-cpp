#include<iostream>
#include<vector>
#include<cmath>


float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x){
    return x * (1.0f - x);
}

struct NeuralNetwork{
    
    
    std::vector <float> weights;
    float bias;

    NeuralNetwork(int input){
        weights.resize(input);
        for(auto &w: weights){
            w = ((rand()%2000)/1000.0f-1.0f);
            bias = ((rand()%2000)/1000.0f-1.0f);

            }

        }
    float activate(const std::vector<float> &a){
        float z = bias;
        for( int i=0;i<weights.size();i++){
            z+=weights[i] * a[i];
        }
        return sigmoid(z);
    }

    
};



int main(){
 srand(time(nullptr));

 NeuralNetwork h1(2),h2(2);

 NeuralNetwork out(2);

 std::vector<std::vector<float>> input = {
         {0, 0}, {0, 1}, {1, 0}, {1, 1}
 };
 std::vector<float> labels = {0,1,1,0};

 float lr =0.1f;
 for(int epoch=0;epoch<10000;++epoch){
    float loss=0;
    for(int i=0;i<4;++i){
        std::vector<float> node= input[i];
        float label = labels[i];

        float o1 = h1.activate(node);
        float o2 = h2.activate(node);

        std::vector<float> hidden = {o1,o2};
        float y_hat = out.activate(hidden);
        float error = label-y_hat;
        loss+=error*error;

        //backpropagation
        float out_grad = error * sigmoid_derivative(y_hat);

        for(int j=0;j<out.weights.size();++j){
            out.weights[j] += lr * out_grad * hidden[j];
            out.bias += lr * out_grad;

            

        }



    }

 }

}
