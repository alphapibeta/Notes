import numpy as np
from pycuda import gpuarray
from cuda_kernels import exp_ker, mean_ker

class SoftmaxLayer:
    def __init__(self, num=None, stream=None):
        self.num = np.int32(num)
        self.stream = stream

    def eval_(self, x, y=None, batch_size=None, stream=None):
        if stream is None:
            stream = self.stream

        if not isinstance(x, gpuarray.GPUArray):
            x = gpuarray.to_gpu_async(np.array(x, dtype=np.float32), stream=stream)

        batch_size = np.int32(batch_size if batch_size is not None else (x.shape[0] if len(x.shape) >= 2 else 1))

        if y is None:
            y_shape = (batch_size, self.num) if batch_size > 1 else (self.num,)
            y = gpuarray.empty(y_shape, dtype=np.float32)

        exp_ker(self.num, x, y, batch_size,
                block=(32, 1, 1), grid=(int(np.ceil(self.num / 32)), 1, 1), stream=stream)

        mean_ker(self.num, y, y, batch_size,
                 block=(32, 1, 1), grid=(int(np.ceil(batch_size / 32)), 1, 1), stream=stream)
        return y