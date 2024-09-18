import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv
from queue import Queue
from layers import DenseLayer, SoftmaxLayer, AttentionLayer
from utils import cross_entropy
import json
import onnx
from onnx import helper, TensorProto, numpy_helper




class SequentialNetwork:
    """
    This class represents a Sequential Neural Network with various functionalities such as adding layers, making predictions, training using batch stochastic gradient descent, exporting the network structure to JSON, loading a network from a JSON file, getting detailed information about each layer, and exporting the network to ONNX format.
    """
    def __init__(self, layers=None, delta=None, stream=None, max_batch_size=32, max_streams=10, epochs=10):
        self.network = []
        self.network_summary = []
        self.network_mem = []
        self.layer_names = []  # store names of layers
        self.stream = stream if stream is not None else drv.Stream() #CUDa stream
        self.delta = delta if delta is not None else 0.0001 #delata rate of learning
        self.max_batch_size = max_batch_size # stacically defined
        self.max_streams = max_streams #cuda streams 
        self.epochs = epochs 
        self.layer_count = 0  # count for naming purposes

        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

    # Generate a unique name for the layer
    def _generate_layer_name(self, layer_type):
        """Generate a unique name for each layer."""
        self.layer_count += 1
        return f"{layer_type}_{self.layer_count}"






    def add_layer(self, layer):
        """Add a layer to the network with proper naming and tracking"""
        layer_type = layer['type']
        layer_name = self._generate_layer_name(layer_type)  

        if layer_type == 'dense':
            num_inputs = layer['num_inputs'] if len(self.network) == 0 else self.network_summary[-1][2]
            num_outputs = layer['num_outputs']
            sigmoid = layer['sigmoid']
            relu = layer['relu']
            weights = layer['weights']
            b = layer['bias']

            # Create Dense Layer and add it to the network
            dense_layer = DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs,
                                    sigmoid=sigmoid, relu=relu, weights=weights, b=b)
            self.network.append(dense_layer)  
            self.network_summary.append(('dense', num_inputs, num_outputs))
            self.layer_names.append(layer_name)  

            # Allocate memory for this layer's input/output
            if self.max_batch_size > 1:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((self.max_batch_size, num_inputs), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((self.max_batch_size, num_outputs), dtype=np.float32))
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((num_inputs,), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((num_outputs,), dtype=np.float32))

        elif layer_type == 'softmax':
            if len(self.network) == 0:
                raise Exception("Error! Softmax layer can't be first!")

            if self.network_summary[-1][0] != 'dense':
                raise Exception("Error! Need a dense layer before a softmax layer!")

            num = self.network_summary[-1][2]
            softmax_layer = SoftmaxLayer(num=num)
            self.network.append(softmax_layer)
            self.network_summary.append(('softmax', num, num))
            self.layer_names.append(layer_name)  # Track the layer name

            # Allocate memory for softmax output
            if self.max_batch_size > 1:
                self.network_mem.append(gpuarray.empty((self.max_batch_size, num), dtype=np.float32))
            else:
                self.network_mem.append(gpuarray.empty((num,), dtype=np.float32))

        elif layer_type == 'attention':
            num_inputs = layer['num_inputs'] if len(self.network) == 0 else self.network_summary[-1][2]
            num_outputs = layer['num_outputs']
            num_heads = layer.get('num_heads', 1)

            # Create Attention Layer and add it to the network
            attention_layer = AttentionLayer(num_inputs=num_inputs, num_outputs=num_outputs, num_heads=num_heads)
            self.network.append(attention_layer)
            self.network_summary.append(('attention', num_inputs, num_outputs))
            self.layer_names.append(layer_name)  # Track the layer name

            # Allocate memory for attention output
            if self.max_batch_size > 1:
                self.network_mem.append(gpuarray.empty((self.max_batch_size, num_outputs), dtype=np.float32))
            else:
                self.network_mem.append(gpuarray.empty((num_outputs,), dtype=np.float32))

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

                
    
    
    def predict(self, x, stream=None):
        if stream is None:
            stream = self.stream

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        input_shape = self.network_mem[0].shape
        if x.shape[1] != input_shape[1]:
            raise ValueError(f"Input shape mismatch! Expected input with {input_shape[1]} features, but got {x.shape[1]}.")

        if x.shape[0] > self.network_mem[0].shape[0]:
            raise ValueError("Batch size exceeds max_batch_size of the network.")

        x0 = np.zeros(self.network_mem[0].shape, dtype=np.float32)
        x0[:x.shape[0], ...] = x
        self.network_mem[0].set_async(x0, stream=stream)

        batch_size = x.shape[0]

        for i in range(len(self.network)):
            self.network[i].eval_(x=self.network_mem[i], y=self.network_mem[i + 1],
                                batch_size=batch_size, stream=stream)

        y = self.network_mem[-1].get_async(stream=stream)
        y = y[:batch_size, ...]
        return y
    
    

    def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None,
                        stream=None, batch_size=None, delta=None):
        self.network[layer_index].eval_(x=self.network_mem[layer_index],
                                        y=partial_mem[layer_index + 1], batch_size=batch_size,
                                        stream=stream, w_t=w_t, b_t=b_t, delta=delta)

        for i in range(layer_index + 1, len(self.network)):
            self.network[i].eval_(x=partial_mem[i], y=partial_mem[i + 1],
                                  batch_size=batch_size, stream=stream)

    def bsgd(self, training=None, labels=None, delta=None, max_streams=None,
            batch_size=None, epochs=1, training_rate=0.01):
        
        """
        This method implements a batch stochastic gradient descent (bsgd) algorithm for training a neural network.
        @param training - The training data
        @param labels - The corresponding labels for the training data
        @param delta - The value used for calculating gradients
        @param max_streams - The maximum number of streams to use
        @param batch_size - The size of each batch
        @param epochs - The number of epochs to train for
        @param training_rate - The learning rate for training the network
        @return None
        """
        
        
        training_rate = np.float32(training_rate)
        training = np.float32(training)
        labels = np.float32(labels)

        if training.shape[0] != labels.shape[0]:
            raise Exception("Number of training data points should be same as labels!")

        max_streams = max_streams if max_streams is not None else self.max_streams
        epochs = epochs if epochs is not None else self.epochs
        delta = delta if delta is not None else self.delta

        streams = []
        bgd_mem = []

        for _ in range(max_streams):
            streams.append(drv.Stream())
            bgd_mem.append([])

        for i in range(len(bgd_mem)):
            for mem_bank in self.network_mem:
                bgd_mem[i].append(gpuarray.empty_like(mem_bank))

        num_points = training.shape[0]
        batch_size = batch_size if batch_size is not None else self.max_batch_size
        index = list(range(num_points))

        for k in range(epochs):
            
            print('-----------------------------------------------------------')
            print(f'Starting training epoch: {k}')
            print(f'Batch size: {batch_size} , Total number of training samples: {num_points}')
            print('-----------------------------------------------------------')

            all_grad = []
            np.random.shuffle(index)


            for r in range(int(np.ceil(training.shape[0] / batch_size))):  # Using np.ceil to ensure all batches are processed
                batch_index = index[r * batch_size:(r + 1) * batch_size]
                batch_training = training[batch_index, :]
                batch_labels = labels[batch_index, :]

                # Adjust batch_size for the last batch
                current_batch_size = batch_training.shape[0]

                batch_predictions = self.predict(batch_training)
                cur_entropy = cross_entropy(predictions=batch_predictions, ground_truth=batch_labels)

                print(f'entropy: {cur_entropy}')

                for i in range(len(self.network)):
                    if self.network_summary[i][0] != 'dense':
                        continue

                    all_weights = Queue()
                    grad_w = np.zeros((self.network[i].weights.size,), dtype=np.float32)
                    grad_b = np.zeros((self.network[i].b.size,), dtype=np.float32)

                    for w in range(self.network[i].weights.size):
                        all_weights.put(('w', np.int32(w)))

                    for b in range(self.network[i].b.size):
                        all_weights.put(('b', np.int32(b)))

                    while not all_weights.empty():
                        stream_weights = Queue()
                        for j in range(max_streams):
                            if all_weights.empty():
                                break

                            wb = all_weights.get()
                            if wb[0] == 'w':
                                w_t = wb[1]
                                b_t = None
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                w_t = None

                            stream_weights.put(wb)
                            self.partial_predict(layer_index=i, w_t=w_t, b_t=b_t, partial_mem=bgd_mem[j],
                                                 stream=streams[j], batch_size=current_batch_size, delta=delta)

                        for j in range(max_streams):
                            if stream_weights.empty():
                                break

                            wb = stream_weights.get()
                            w_predictions = bgd_mem[j][-1].get_async(stream=streams[j])
                            w_predictions = w_predictions[:current_batch_size, ...]  # Adjust for current batch size
                            w_entropy = cross_entropy(predictions=w_predictions, ground_truth=batch_labels)

                            if wb[0] == 'w':
                                w_t = wb[1]
                                grad_w[w_t] = -(w_entropy - cur_entropy) / delta
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                grad_b[b_t] = -(w_entropy - cur_entropy) / delta

                    all_grad.append([np.reshape(grad_w, self.network[i].weights.shape), grad_b])

            grad_index = 0
            for i in range(len(self.network)):
                if self.network_summary[i][0] == 'dense':
                    new_weights = self.network[i].weights.get()
                    new_weights += training_rate * all_grad[grad_index][0]
                    new_bias = self.network[i].b.get()
                    new_bias += training_rate * all_grad[grad_index][1]
                    self.network[i].weights.set(new_weights)
                    self.network[i].b.set(new_bias)
                    grad_index += 1

    
    
    def export_network(self, filename):
        """Export the network structure and weights to a JSON file."""
        network_data = {
            "layers": [],
            "delta": self.delta,
            "max_batch_size": self.max_batch_size,
            "max_streams": self.max_streams,
            "epochs": self.epochs
        }

        for i, layer in enumerate(self.network):
            layer_data = {
                "type": self.network_summary[i][0],
                "num_inputs": self.network_summary[i][1],
                "num_outputs": self.network_summary[i][2]
            }

            if isinstance(layer, DenseLayer):
                layer_data.update({
                    "weights": layer.weights.get().tolist(),
                    "bias": layer.b.get().tolist(),
                    "relu": bool(layer.relu),
                    "sigmoid": bool(layer.sigmoid)
                })
            elif isinstance(layer, AttentionLayer):
                layer_data.update({
                    "num_heads": layer.num_heads
                })

            network_data["layers"].append(layer_data)

        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2)

    @classmethod
    def load_network(cls, filename):
        """Load a network from a JSON file."""
        with open(filename, 'r') as f:
            network_data = json.load(f)

        network = cls(
            delta=network_data["delta"],
            max_batch_size=network_data["max_batch_size"],
            max_streams=network_data["max_streams"],
            epochs=network_data["epochs"]
        )

        for layer_data in network_data["layers"]:
            if layer_data["type"] == "dense":
                network.add_layer({
                    "type": "dense",
                    "num_inputs": layer_data["num_inputs"],
                    "num_outputs": layer_data["num_outputs"],
                    "relu": layer_data["relu"],
                    "sigmoid": layer_data["sigmoid"],
                    "weights": np.array(layer_data["weights"]),
                    "bias": np.array(layer_data["bias"])
                })
            elif layer_data["type"] == "attention":
                network.add_layer({
                    "type": "attention",
                    "num_inputs": layer_data["num_inputs"],
                    "num_outputs": layer_data["num_outputs"],
                    "num_heads": layer_data["num_heads"]
                })
            elif layer_data["type"] == "softmax":
                network.add_layer({"type": "softmax"})

        return network

    def get_layer_details(self):
        """Return detailed information about each layer in the network."""
        details = []
        for i, layer in enumerate(self.network):
            layer_info = {
                "name": self.layer_names[i],  
                "type": self.network_summary[i][0],
                "num_inputs": self.network_summary[i][1],
                "num_outputs": self.network_summary[i][2]
            }
            if isinstance(layer, DenseLayer):
                layer_info.update({
                    "activation": "ReLU" if layer.relu else ("Sigmoid" if layer.sigmoid else "None"),
                    "weights_shape": layer.weights.shape,
                    "bias_shape": layer.b.shape
                })
            elif isinstance(layer, SoftmaxLayer):
                layer_info.update({
                    "activation": "Softmax",
                    "num_inputs": self.network_summary[i][1],
                    "num_outputs": self.network_summary[i][2]
                })
            elif isinstance(layer, AttentionLayer):
                layer_info.update({
                    "num_heads": layer.num_heads
                })
            details.append(layer_info)
        return details

    


    def export_to_onnx(self, filename, input_shape):
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output_shape = [input_shape[0], self.network_summary[-1][2]]
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        onnx_initializers = []
        onnx_nodes = []
        prev_output = 'input'

        print(f"Starting with input: {prev_output}")

        for i, layer in enumerate(self.network):
            layer_name = self.layer_names[i]
            print(f"\nProcessing layer: {layer_name}, type: {type(layer).__name__}")

            if isinstance(layer, DenseLayer):
                weights = layer.weights.get().T
                bias = layer.b.get()

                bias = bias.reshape(-1)

                print(f"  Layer: {layer_name}, Weight shape: {weights.shape}, Bias shape: {bias.shape}")

                weight_tensor = helper.make_tensor(
                    name=f'{layer_name}_W',
                    data_type=TensorProto.FLOAT,
                    dims=weights.shape,
                    vals=weights.flatten().tolist()
                )
                
                bias_tensor = helper.make_tensor(
                    name=f'{layer_name}_b',
                    data_type=TensorProto.FLOAT,
                    dims=bias.shape,
                    vals=bias.tolist()
                )

                node = helper.make_node(
                    'Gemm',
                    inputs=[prev_output, f'{layer_name}_W', f'{layer_name}_b'],
                    outputs=[f'{layer_name}_output'],
                    name=layer_name,
                    alpha=1.0,
                    beta=1.0,
                    transB=1
                )
                onnx_nodes.append(node)
                print(f"  Added Gemm node: inputs={node.input}, outputs={node.output}")

                onnx_initializers.extend([weight_tensor, bias_tensor])
                prev_output = f'{layer_name}_output'

                if layer.relu:
                    node = helper.make_node(
                        'Relu',
                        inputs=[prev_output],
                        outputs=[f'{layer_name}_relu_output'],
                        name=f'{layer_name}_relu'
                    )
                    onnx_nodes.append(node)
                    print(f"  Added Relu node: inputs={node.input}, outputs={node.output}")
                    prev_output = f'{layer_name}_relu_output'
                elif layer.sigmoid:
                    node = helper.make_node(
                        'Sigmoid',
                        inputs=[prev_output],
                        outputs=[f'{layer_name}_sigmoid_output'],
                        name=f'{layer_name}_sigmoid'
                    )
                    onnx_nodes.append(node)
                    print(f"  Added Sigmoid node: inputs={node.input}, outputs={node.output}")
                    prev_output = f'{layer_name}_sigmoid_output'

            elif isinstance(layer, SoftmaxLayer):
                softmax_node = helper.make_node(
                    'Softmax',
                    inputs=[prev_output],
                    outputs=['output'],  
                    axis=1,
                    name=f'{layer_name}_softmax'
                )
                onnx_nodes.append(softmax_node)
                print(f"  Added Softmax node: inputs={softmax_node.input}, outputs={softmax_node.output}")
                prev_output = 'output'

            print(f"  Layer output: {prev_output}")

        print("\nCreating ONNX graph...")
        graph_def = helper.make_graph(
            onnx_nodes,
            "SequentialNetworkGraph",
            [input_tensor],
            [output_tensor],  
            initializer=onnx_initializers
        )

        print("Creating ONNX model...")
        onnx_model = helper.make_model(graph_def, producer_name='sequential_network')

        print("Checking ONNX model...")
        try:
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid.")
        except onnx.checker.ValidationError as e:
            print(f"ONNX model is invalid: {str(e)}")
            return

        print(f"Saving ONNX model to {filename}...")
        onnx.save(onnx_model, filename)
        print(f"Model exported to {filename}")