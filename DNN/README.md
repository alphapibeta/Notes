# Iris Classifier Neural Network

## Project Overview

This project implements a custom neural network framework to classify Iris flowers. It's built from scratch using Python and CUDA, demonstrating how to create, train, and export neural networks for machine learning tasks.

## Key Features

- Custom neural network implementation with CUDA acceleration
- Supports dense layers, softmax, and attention mechanisms
- Trains on the Iris dataset for flower classification
- Exports trained models to ONNX format for interoperability

## Project Structure

### Main Components

1. `iris_train_main.py`: The entry point of the application. It loads the Iris dataset, creates and trains the network, and exports it to ONNX.

2. `network/sequential_network.py`: Defines the `SequentialNetwork` class, which is the core of our neural network architecture.

3. `layers/`: Contains implementations of different layer types:
   - `dense_layer.py`: Dense (fully connected) layer
   - `softmax_layer.py`: Softmax activation layer
   - `attention_layer.py`: Attention mechanism layer

4. `cuda_kernels/`: CUDA implementations for faster computations:
   - `dense_eval.py`: CUDA kernels for dense layer operations
   - `softmax.py`: CUDA kernels for softmax computations
   - `attention.py`: CUDA kernels for attention mechanisms

5. `utils/`: Helper functions and utilities:
   - `cross_entropy.py`: Implements the cross-entropy loss function
   - `data_processing.py`: Functions for data normalization and preprocessing

6. `onnx_load.py`: A script to load and verify the exported ONNX model

## How It Works

1. Data Loading: The Iris dataset is loaded and split into training and testing sets.
2. Network Creation: A sequential network is created with dense layers and a softmax output.
3. Training: The network is trained using batch stochastic gradient descent (BSGD).
4. Evaluation: The trained network is tested on the holdout set.
5. Export: The trained model is exported to ONNX format for use in other frameworks.

## Getting Started

1. Ensure you have Python 3.x and CUDA installed on your system.
2. Install required packages: `numpy`, `pycuda`, `onnx`, `onnxruntime`.
3. Run `python iris_train_main.py` to train the model and export it.
4. Use `python onnx_load.py` to verify the exported ONNX model.

## Customization

You can modify the network architecture in `iris_train_main.py` by adding or removing layers. The `SequentialNetwork` class supports easy addition of new layer types.



python3 iris_train_main.py
```

Step 1: Loading Iris dataset
Loaded 150 samples

Step 2: Shuffling and splitting dataset
Training set size: 100, Test set size: 50

Step 3: Creating Sequential Network
Network structure:
  Layer 0: {'name': 'dense_1', 'type': 'dense', 'num_inputs': 4, 'num_outputs': 32, 'activation': 'ReLU', 'weights_shape': (32, 4), 'bias_shape': (32,)}
  Layer 1: {'name': 'dense_2', 'type': 'dense', 'num_inputs': 32, 'num_outputs': 10, 'activation': 'ReLU', 'weights_shape': (10, 32), 'bias_shape': (10,)}
  Layer 2: {'name': 'dense_3', 'type': 'dense', 'num_inputs': 10, 'num_outputs': 3, 'activation': 'None', 'weights_shape': (3, 10), 'bias_shape': (3,)}
  Layer 3: {'name': 'softmax_4', 'type': 'softmax', 'num_inputs': 3, 'num_outputs': 3, 'activation': 'Softmax'}

Step 4: Conditioning training data
Data conditioned. Mean: [5.859999  3.045     3.7869995 1.2039999], Std: [0.8485281  0.4196129  1.7774506  0.75735325]

Step 5: Training the network
-----------------------------------------------------------
Starting training epoch: 0
Batch size: 32 , Total number of training samples: 100
-----------------------------------------------------------
entropy: 0.5784383318563647
entropy: 0.5738383269381818
entropy: 0.5789436461019308
entropy: 0.566995896238285
Training completed in 2.78 seconds

Step 6: Exporting trained network
Network exported to iris_network.json

Step 7: Testing the network
Test Accuracy: 38.00%
Total Training Time: 2.78 seconds

Step 8: Loading exported network and verifying
Loaded network structure:
  Layer 0: {'name': 'dense_1', 'type': 'dense', 'num_inputs': 4, 'num_outputs': 32, 'activation': 'ReLU', 'weights_shape': (32, 4), 'bias_shape': (32,)}
  Layer 1: {'name': 'dense_2', 'type': 'dense', 'num_inputs': 32, 'num_outputs': 10, 'activation': 'ReLU', 'weights_shape': (10, 32), 'bias_shape': (10,)}
  Layer 2: {'name': 'dense_3', 'type': 'dense', 'num_inputs': 10, 'num_outputs': 3, 'activation': 'None', 'weights_shape': (3, 10), 'bias_shape': (3,)}
  Layer 3: {'name': 'softmax_4', 'type': 'softmax', 'num_inputs': 3, 'num_outputs': 3, 'activation': 'Softmax'}
Loaded Network Test Accuracy: 38.00%

Step 9: Exporting network to ONNX format
Starting with input: input

Processing layer: dense_1, type: DenseLayer
  Layer: dense_1, Weight shape: (4, 32), Bias shape: (32,)
  Added Gemm node: inputs=['input', 'dense_1_W', 'dense_1_b'], outputs=['dense_1_output']
  Added Relu node: inputs=['dense_1_output'], outputs=['dense_1_relu_output']
  Layer output: dense_1_relu_output

Processing layer: dense_2, type: DenseLayer
  Layer: dense_2, Weight shape: (32, 10), Bias shape: (10,)
  Added Gemm node: inputs=['dense_1_relu_output', 'dense_2_W', 'dense_2_b'], outputs=['dense_2_output']
  Added Relu node: inputs=['dense_2_output'], outputs=['dense_2_relu_output']
  Layer output: dense_2_relu_output

Processing layer: dense_3, type: DenseLayer
  Layer: dense_3, Weight shape: (10, 3), Bias shape: (3,)
  Added Gemm node: inputs=['dense_2_relu_output', 'dense_3_W', 'dense_3_b'], outputs=['dense_3_output']
  Layer output: dense_3_output

Processing layer: softmax_4, type: SoftmaxLayer
  Added Softmax node: inputs=['dense_3_output'], outputs=['output']
  Layer output: output

Creating ONNX graph...
Creating ONNX model...
Checking ONNX model...
ONNX model is valid.
Saving ONNX model to iris_network.onnx...
Model exported to iris_network.onnx
Network exported to iris_network.onnx

Debug: Network structure
Layer 0: {'name': 'dense_1', 'type': 'dense', 'num_inputs': 4, 'num_outputs': 32, 'activation': 'ReLU', 'weights_shape': (32, 4), 'bias_shape': (32,)}
Layer 1: {'name': 'dense_2', 'type': 'dense', 'num_inputs': 32, 'num_outputs': 10, 'activation': 'ReLU', 'weights_shape': (10, 32), 'bias_shape': (10,)}
Layer 2: {'name': 'dense_3', 'type': 'dense', 'num_inputs': 10, 'num_outputs': 3, 'activation': 'None', 'weights_shape': (3, 10), 'bias_shape': (3,)}
Layer 3: {'name': 'softmax_4', 'type': 'softmax', 'num_inputs': 3, 'num_outputs': 3, 'activation': 'Softmax'}

```

