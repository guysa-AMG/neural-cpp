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
        std::vector<float> values= input[i];
        float label = labels[i];

        float o1 = h1.activate(values);
        float o2 = h2.activate(values);

        std::vector<float> hidden = {o1,o2};
        float y_hat = out.activate(hidden);
        float error = label-y_hat;
        loss+=error*error;

        //backpropagation
        float out_grad = error * sigmoid_derivative(y_hat);

        for(int j=0;j<out.weights.size();++j)
            out.weights[j] += lr * out_grad * hidden[j];
        out.bias += lr * out_grad;

        float h1_grad = out.weights[0] * out_grad * sigmoid_derivative(o1);
        float h2_grad = out.weights[1] * out_grad * sigmoid_derivative(o2);

        for(int k=0;k<values.size();++k){
                h1.weights[k] += lr * h1_grad * values[k];
                h2.weights[k] += lr *h2_grad * values[k];

            }
            h1.bias += lr * h1_grad;
            h2.bias += lr * h2_grad;
        }
        if(epoch%1000==0)
            std::cout<<"epoch: "<<epoch<<" loss: "<<loss<<std::endl;
        }

        std::cout<<"\n Final prediction\n";

        for(int j=0;j<4;++j){
            float o1 = h1.activate(input[j]);
            float o2 = h2.activate(input[j]);
            float output = out.activate({o1,o2});

            std::cout<<input[j][0]<<" , "<<input[j][1]<<"-->"<<output<<std::endl;
        }



    }

 


