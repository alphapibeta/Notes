try:
    import hessian_inversion_cpu
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False

try:
    import hessian_inversion_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class HessianInversion:
    def __init__(self, size, use_gpu=True, use_double=False):
        if use_gpu and GPU_AVAILABLE:
            # GPU initialization
            if use_double:
                self._impl = hessian_inversion_gpu.HessianInversionGPUDouble(size)
            else:
                self._impl = hessian_inversion_gpu.HessianInversionGPUFloat(size)
        elif CPU_AVAILABLE:
            # CPU initialization
            if use_double:
                self._impl = hessian_inversion_cpu.HessianInversionCPUDouble(size)
            else:
                self._impl = hessian_inversion_cpu.HessianInversionCPUFloat(size)
        else:
            raise ImportError("Neither CPU nor GPU implementation is available")

    def setMatrix(self, matrix):
        return self._impl.setMatrix(matrix)

    def getInverse(self):
        return self._impl.getInverse()

    def invert(self):
        return self._impl.invert()

    def regularizeMatrix(self, epsilon=1e-6):
        return self._impl.regularizeMatrix(epsilon)

    # GPU-specific methods, only available if the GPU implementation is used
    def getGPUMemoryThroughput(self):
        if hasattr(self._impl, 'getGPUMemoryThroughput'):
            return self._impl.getGPUMemoryThroughput()
        raise AttributeError("This method is only available for GPU implementations")

    def getGPUComputationalThroughput(self):
        if hasattr(self._impl, 'getGPUComputationalThroughput'):
            return self._impl.getGPUComputationalThroughput()
        raise AttributeError("This method is only available for GPU implementations")

    def getArithmeticIntensity(self):
        if hasattr(self._impl, 'getArithmeticIntensity'):
            return self._impl.getArithmeticIntensity()
        raise AttributeError("This method is only available for GPU implementations")

    def getGPUBandwidth(self):
        if hasattr(self._impl, 'getGPUBandwidth'):
            return self._impl.getGPUBandwidth()
        raise AttributeError("This method is only available for GPU implementations")

    def getTheoreticalGPUBandwidth(self):
        if hasattr(self._impl, 'getTheoreticalGPUBandwidth'):
            return self._impl.getTheoreticalGPUBandwidth()
        raise AttributeError("This method is only available for GPU implementations")

    def getTheoreticalGPUComputationalThroughput(self):
        if hasattr(self._impl, 'getTheoreticalGPUComputationalThroughput'):
            return self._impl.getTheoreticalGPUComputationalThroughput()
        raise AttributeError("This method is only available for GPU implementations")
