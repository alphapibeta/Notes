import numpy as np
from pycuda import gpuarray
from cuda_kernels import compute_attention_scores_ker, apply_softmax_ker, compute_attention_output_ker
from .dense_layer import DenseLayer

class AttentionLayer:
    def __init__(self, num_inputs=None, num_outputs=None, num_heads=1, stream=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_heads = num_heads
        self.stream = stream

        self.W_Q = DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs)
        self.W_K = DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs)
        self.W_V = DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs)
        self.W_O = DenseLayer(num_inputs=num_outputs, num_outputs=num_outputs)

    def eval_(self, x, y=None, batch_size=None, stream=None):
        if stream is None:
            stream = self.stream

        if not isinstance(x, gpuarray.GPUArray):
            x = gpuarray.to_gpu_async(np.array(x, dtype=np.float32), stream=stream)

        if x.ndim == 2:
            # Assume sequence_length=1
            batch_size = x.shape[0]
            sequence_length = 1
            embedding_size = x.shape[1]
            x = x.reshape((batch_size, sequence_length, embedding_size))
        else:
            batch_size, sequence_length, embedding_size = x.shape

        # Reshape x to (batch_size * sequence_length, embedding_size)
        x_reshaped = x.reshape((batch_size * sequence_length, embedding_size))

        # Compute Q, K, V
        Q = self.W_Q.eval_(x_reshaped, batch_size=batch_size * sequence_length, stream=stream)
        K = self.W_K.eval_(x_reshaped, batch_size=batch_size * sequence_length, stream=stream)
        V = self.W_V.eval_(x_reshaped, batch_size=batch_size * sequence_length, stream=stream)

        # Reshape Q, K, V to (batch_size, sequence_length, num_outputs)
        Q = Q.reshape((batch_size, sequence_length, self.num_outputs))
        K = K.reshape((batch_size, sequence_length, self.num_outputs))
        V = V.reshape((batch_size, sequence_length, self.num_outputs))

        d_k = self.num_outputs

        # Allocate memory for attention scores and output
        attention_scores = gpuarray.empty((batch_size, sequence_length, sequence_length), dtype=np.float32)
        attention_output = gpuarray.empty((batch_size, sequence_length, self.num_outputs), dtype=np.float32)

        # Compute attention scores
        block_dim = (16, 16, 1)
        grid_dim = (
            (sequence_length + block_dim[0] - 1) // block_dim[0],
            (sequence_length + block_dim[1] - 1) // block_dim[1],
            batch_size
        )
        compute_attention_scores_ker(
            Q, K, attention_scores,
            np.int32(batch_size), np.int32(sequence_length), np.int32(d_k),
            block=block_dim, grid=grid_dim, stream=stream
        )

        # Apply softmax to attention scores
        block_dim_softmax = (32, 1, 1)
        grid_dim_softmax = (
            1,
            (sequence_length + block_dim_softmax[0] - 1) // block_dim_softmax[0],
            batch_size
        )
        apply_softmax_ker(
            attention_scores,
            np.int32(batch_size), np.int32(sequence_length),
            block=block_dim_softmax, grid=grid_dim_softmax, stream=stream
        )

        # Compute attention output
        block_dim_output = (16, 16, 1)
        grid_dim_output = (
            (self.num_outputs + block_dim_output[0] - 1) // block_dim_output[0],
            (sequence_length + block_dim_output[1] - 1) // block_dim_output[1],
            batch_size
        )
        compute_attention_output_ker(
            attention_scores, V, attention_output,
            np.int32(batch_size), np.int32(sequence_length), np.int32(self.num_outputs),
            block=block_dim_output, grid=grid_dim_output, stream=stream
        )

        # Reshape attention output to (batch_size * sequence_length, num_outputs)
        attention_output_reshaped = attention_output.reshape((batch_size * sequence_length, self.num_outputs))

        # Output projection
        y_out = self.W_O.eval_(attention_output_reshaped, batch_size=batch_size * sequence_length, stream=stream)
        y_out = y_out.reshape((batch_size, sequence_length, self.num_outputs))

        # Flatten if sequence_length == 1
        if sequence_length == 1:
            y_out = y_out.reshape((batch_size, self.num_outputs))

        if y is not None:
            y.set_async(y_out.astype(np.float32), stream=stream)
        else:
            y = y_out

        return y
    def get_onnx_attributes(self):
        return {
            "num_heads": self.num_heads,
            "W_Q": self.W_Q.weights,
            "W_K": self.W_K.weights,
            "W_V": self.W_V.weights,
            "W_O": self.W_O.weights
        }