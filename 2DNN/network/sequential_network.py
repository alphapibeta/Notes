import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv
from queue import Queue
from layers import DenseLayer, SoftmaxLayer, AttentionLayer
from utils import cross_entropy

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