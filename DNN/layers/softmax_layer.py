import numpy as np
from pycuda import gpuarray
from cuda_kernels import exp_ker, mean_ker

class SoftmaxLayer:
    """
    A class representing a Softmax layer with methods to initialize the layer and get the ONNX node for export.
    @param num - The number of classes for the Softmax layer.
    @param stream - The stream for the Softmax layer.
    Method:
    - __init__(self, num=None, stream=None): Initializes the Softmax layer with the given number of classes.
    - get_onnx_node(self, input_name, output_name): Create and return the ONNX Softmax node for export.
    """

    
    def __init__(self, num=None, stream=None):
        self.num = np.int32(num)
        self.stream = stream

    def eval_(self, x, y=None, batch_size=None, stream=None):
        """
        Evaluates the softmax over the input x, which is the output of the last dense layer.
        """
        if stream is None:
            stream = self.stream

        # Ensure the input is a GPUArray
        if not isinstance(x, gpuarray.GPUArray):
            x = gpuarray.to_gpu_async(np.array(x, dtype=np.float32), stream=stream)

        # Determine the batch size
        batch_size = np.int32(batch_size if batch_size is not None else (x.shape[0] if len(x.shape) >= 2 else 1))

        # Allocate space for the output
        if y is None:
            y_shape = (batch_size, self.num) if batch_size > 1 else (self.num,)
            y = gpuarray.empty(y_shape, dtype=np.float32)

        # Apply the exponential kernel
        exp_ker(self.num, x, y, batch_size,
                block=(32, 1, 1), grid=(int(np.ceil(self.num / 32)), 1, 1), stream=stream)

        # Apply the mean kernel for normalization
        mean_ker(self.num, y, y, batch_size,
                 block=(32, 1, 1), grid=(int(np.ceil(batch_size / 32)), 1, 1), stream=stream)

        return y

    def get_onnx_node(self, input_name, output_name):
        """
        Create and return the ONNX Softmax node for export.
        """
        softmax_node = helper.make_node(
            'Softmax',
            inputs=[input_name],
            outputs=[output_name],
            axis=1  # Assuming softmax applies across the classes (axis=1)
        )
        return softmax_node
