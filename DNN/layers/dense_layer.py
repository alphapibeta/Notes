import numpy as np
from pycuda import gpuarray
from cuda_kernels import eval_ker

class DenseLayer:
    def __init__(self, num_inputs=None, num_outputs=None, weights=None, b=None, stream=None,
        relu=False, sigmoid=False, delta=None):
        """
        Initialize a neural network layer with specified parameters.
        @param num_inputs - number of input neurons
        @param num_outputs - number of output neurons
        @param weights - weights for the layer
        @param b - bias for the layer
        @param stream - CUDA stream for GPU operations
        @param relu - whether to apply ReLU activation
        @param sigmoid - whether to apply sigmoid activation
        @param delta - learning rate
        @return None
        """
        
        
        
        
        self.stream = stream    
        self.delta = np.float32(delta if delta is not None else 0.001)

        if weights is None:
            weights = (np.random.rand(num_outputs, num_inputs) - 0.5)
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)
        else:
            self.num_inputs = np.int32(weights.shape[1])
            self.num_outputs = np.int32(weights.shape[0])

        if not isinstance(weights, gpuarray.GPUArray):
            self.weights = gpuarray.to_gpu_async(np.array(weights, dtype=np.float32), stream=self.stream)
        else:
            self.weights = weights

        if b is None:
            b = gpuarray.zeros((self.num_outputs,), dtype=np.float32)
        if not isinstance(b, gpuarray.GPUArray):
            self.b = gpuarray.to_gpu_async(np.array(b, dtype=np.float32), stream=self.stream)
        else:
            self.b = b

        self.relu = np.int32(relu)
        self.sigmoid = np.int32(sigmoid)
        self.block = (32, 1, 1)
        self.grid = (int(np.ceil(self.num_outputs / 32)), 1, 1)

    def eval_(self, x, y=None, batch_size=None, stream=None, delta=None, w_t=None, b_t=None):
        if stream is None:
            stream = self.stream

        if not isinstance(x, gpuarray.GPUArray):
            x = gpuarray.to_gpu_async(np.array(x, dtype=np.float32), stream=self.stream)

        batch_size = np.int32(batch_size if batch_size is not None else (x.shape[0] if len(x.shape) >= 2 else 1))
        delta = np.float32(delta if delta is not None else self.delta)
        w_t = np.int32(w_t if w_t is not None else -1)
        b_t = np.int32(b_t if b_t is not None else -1)

        if y is None:
            y_shape = (batch_size, self.num_outputs) if batch_size > 1 else (self.num_outputs,)
            y = gpuarray.empty(y_shape, dtype=np.float32)

        eval_ker(self.num_outputs, self.num_inputs, self.relu, self.sigmoid,
                 self.weights, self.b, x, y, batch_size, w_t, b_t,
                 delta, block=self.block, grid=self.grid, stream=stream)
        return y
    # def get_onnx_attributes(self):
    #     return {
    #         "weights": self.weights.get(),
    #         "bias": self.b.get(),
    #         "relu": self.relu,
    #         "sigmoid": self.sigmoid
    #     }