import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv
from queue import Queue
from layers import DenseLayer, SoftmaxLayer, AttentionLayer
from utils import cross_entropy
import json
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

class SequentialNetwork:
    def __init__(self, layers=None, delta=None, stream=None, max_batch_size=32, max_streams=10, epochs=10):
        self.network = []
        self.network_summary = []
        self.network_mem = []
        self.stream = stream if stream is not None else drv.Stream()
        self.delta = delta if delta is not None else 0.0001
        self.max_batch_size = max_batch_size
        self.max_streams = max_streams
        self.epochs = epochs

        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

    def add_layer(self, layer):
        if layer['type'] == 'dense':
            num_inputs = layer['num_inputs'] if len(self.network) == 0 else self.network_summary[-1][2]
            num_outputs = layer['num_outputs']
            sigmoid = layer['sigmoid']
            relu = layer['relu']
            weights = layer['weights']
            b = layer['bias']

            self.network.append(DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs,
                                           sigmoid=sigmoid, relu=relu, weights=weights, b=b))
            self.network_summary.append(('dense', num_inputs, num_outputs))

            if self.max_batch_size > 1:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((self.max_batch_size, num_inputs), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((self.max_batch_size, num_outputs), dtype=np.float32))
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((num_inputs,), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((num_outputs,), dtype=np.float32))

        elif layer['type'] == 'softmax':
            if len(self.network) == 0:
                raise Exception("Error! Softmax layer can't be first!")

            if self.network_summary[-1][0] != 'dense':
                raise Exception("Error! Need a dense layer before a softmax layer!")

            num = self.network_summary[-1][2]
            self.network.append(SoftmaxLayer(num=num))
            self.network_summary.append(('softmax', num, num))

            if self.max_batch_size > 1:
                self.network_mem.append(gpuarray.empty((self.max_batch_size, num), dtype=np.float32))
            else:
                self.network_mem.append(gpuarray.empty((num,), dtype=np.float32))

        elif layer['type'] == 'attention':
            num_inputs = layer['num_inputs'] if len(self.network) == 0 else self.network_summary[-1][2]
            num_outputs = layer['num_outputs']
            num_heads = layer.get('num_heads', 1)

            self.network.append(AttentionLayer(num_inputs=num_inputs, num_outputs=num_outputs, num_heads=num_heads))
            self.network_summary.append(('attention', num_inputs, num_outputs))

            if self.max_batch_size > 1:
                # Assuming sequence length is provided
                sequence_length = layer.get('sequence_length', 1)
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((self.max_batch_size, num_inputs), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((self.max_batch_size, num_outputs), dtype=np.float32))
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((num_inputs,), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((num_outputs,), dtype=np.float32))

    def predict(self, x, stream=None):
        if stream is None:
            stream = self.stream

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        if x.shape[0] > self.network_mem[0].shape[0]:
            raise Exception("Error: batch size too large for input.")

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

            for r in range(int(np.floor(training.shape[0] / batch_size))):
                batch_index = index[r * batch_size:(r + 1) * batch_size]
                batch_training = training[batch_index, :]
                batch_labels = labels[batch_index, :]

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
                                                 stream=streams[j], batch_size=batch_size, delta=delta)

                        for j in range(max_streams):
                            if stream_weights.empty():
                                break

                            wb = stream_weights.get()
                            w_predictions = bgd_mem[j][-1].get_async(stream=streams[j])
                            w_predictions = w_predictions[:batch_size, ...]
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
            elif isinstance(layer, AttentionLayer):
                layer_info.update({
                    "num_heads": layer.num_heads
                })
            details.append(layer_info)
        return details
    
    
    def export_onnx(self, filename, input_shape):
        inputs = []
        outputs = []
        nodes = []
        initializers = []

        input_name = "input"
        inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape))
        last_output = input_name
        last_shape = input_shape

        for i, layer in enumerate(self.network):
            if isinstance(layer, DenseLayer):
                weight_name = f"weight_{i}"
                bias_name = f"bias_{i}"
                output_name = f"dense_output_{i}"

                weights = layer.weights.get()
                bias = layer.b.get()

                initializers.append(numpy_helper.from_array(weights, name=weight_name))
                initializers.append(numpy_helper.from_array(bias, name=bias_name))

                node = helper.make_node(
                    "Gemm",
                    inputs=[last_output, weight_name, bias_name],
                    outputs=[output_name],
                    name=f"dense_{i}"
                )
                nodes.append(node)

                last_shape = (last_shape[0], weights.shape[0])  # Update shape

                if layer.relu:
                    relu_output = f"relu_output_{i}"
                    relu_node = helper.make_node(
                        "Relu",
                        inputs=[output_name],
                        outputs=[relu_output],
                        name=f"relu_{i}"
                    )
                    nodes.append(relu_node)
                    last_output = relu_output
                elif layer.sigmoid:
                    sigmoid_output = f"sigmoid_output_{i}"
                    sigmoid_node = helper.make_node(
                        "Sigmoid",
                        inputs=[output_name],
                        outputs=[sigmoid_output],
                        name=f"sigmoid_{i}"
                    )
                    nodes.append(sigmoid_node)
                    last_output = sigmoid_output
                else:
                    last_output = output_name

            elif isinstance(layer, SoftmaxLayer):
                softmax_output = f"softmax_output_{i}"
                softmax_node = helper.make_node(
                    "Softmax",
                    inputs=[last_output],
                    outputs=[softmax_output],
                    name=f"softmax_{i}"
                )
                nodes.append(softmax_node)
                last_output = softmax_output

            elif isinstance(layer, AttentionLayer):
                q_weight = f"q_weight_{i}"
                k_weight = f"k_weight_{i}"
                v_weight = f"v_weight_{i}"
                o_weight = f"o_weight_{i}"

                q_weights = layer.W_Q.weights.get()
                k_weights = layer.W_K.weights.get()
                v_weights = layer.W_V.weights.get()
                o_weights = layer.W_O.weights.get()

                initializers.extend([
                    numpy_helper.from_array(q_weights, name=q_weight),
                    numpy_helper.from_array(k_weights, name=k_weight),
                    numpy_helper.from_array(v_weights, name=v_weight),
                    numpy_helper.from_array(o_weights, name=o_weight)
                ])

                q_output = f"q_output_{i}"
                k_output = f"k_output_{i}"
                v_output = f"v_output_{i}"

                nodes.extend([
                    helper.make_node("MatMul", [last_output, q_weight], [q_output], name=f"q_matmul_{i}"),
                    helper.make_node("MatMul", [last_output, k_weight], [k_output], name=f"k_matmul_{i}"),
                    helper.make_node("MatMul", [last_output, v_weight], [v_output], name=f"v_matmul_{i}")
                ])

                # Transpose K
                k_transpose = f"k_transpose_{i}"
                nodes.append(helper.make_node("Transpose", inputs=[k_output], outputs=[k_transpose], name=f"k_transpose_{i}"))

                attention_scores = f"attention_scores_{i}"
                attention_probs = f"attention_probs_{i}"
                attention_output = f"attention_output_{i}"

                scale_value = np.float32(np.sqrt(q_weights.shape[0]))
                scale_name = f"scale_{i}"
                initializers.append(numpy_helper.from_array(np.array(scale_value), name=scale_name))

                nodes.extend([
                    helper.make_node("MatMul", [q_output, k_transpose], [f"qk_matmul_{i}"], name=f"qk_matmul_{i}"),
                    helper.make_node("Div", [f"qk_matmul_{i}", scale_name], [attention_scores], name=f"scale_scores_{i}"),
                    helper.make_node("Softmax", [attention_scores], [attention_probs], name=f"attention_softmax_{i}"),
                    helper.make_node("MatMul", [attention_probs, v_output], [attention_output], name=f"attention_output_{i}")
                ])

                final_output = f"final_output_{i}"
                nodes.append(helper.make_node("MatMul", [attention_output, o_weight], [final_output], name=f"output_projection_{i}"))

                last_output = final_output
                last_shape = (last_shape[0], o_weights.shape[0])  # Update shape

        outputs.append(helper.make_tensor_value_info(last_output, TensorProto.FLOAT, last_shape))

        graph = helper.make_graph(
            nodes, "SequentialNetwork", inputs, outputs, initializers
        )

        model = helper.make_model(graph)
        onnx.save(model, filename)