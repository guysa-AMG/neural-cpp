# 🧠 C++ Neural Network Playground

A collection of machine learning experiments written entirely in modern C++—built from scratch for performance, transparency, and educational clarity. No frameworks. No dependencies. Just handcrafted logic that teaches how ML works under the hood.

## 🚀 Included Projects

| Module            | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| `xor_net.cpp`     | Feedforward NN solving the XOR problem with backpropagation     |
| `activations.hpp` | Sigmoid, ReLU, Tanh activation functions                        |
| `matrix.hpp`      | Minimal matrix math utility for neural computation              |
| `trainer.cpp`     | Training loop with gradient descent and loss tracking           |

> Additional C++ modules are in progress (e.g. voice embedding, OpenCV classifiers, audio buffers)

## 🛠️ Build & Run

Compile using any C++17 or newer compiler:

```bash
g++ xor_net.cpp -o xor_net -std=c++17
./xor_net
