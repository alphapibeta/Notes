from pycuda.compiler import SourceModule

SoftmaxExpCode = '''
__global__ void softmax_exp(int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num)
    {
        for (int k = 0; k < batch_size; k++)
        {
            y[num * k + i] = expf(x[num * k + i]);
        }
    }
}
'''

SoftmaxMeanCode = '''
__global__ void softmax_mean(int num, float *x, float *y, int batch_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < batch_size)
    {
        float temp = 0.0f;

        for (int k = 0; k < num; k++)
            temp += x[i * num + k];

        for (int k = 0; k < num; k++)
            y[i * num + k] = x[i * num + k] / temp;
    }
    return;
}
'''

softmax_exp_mod = SourceModule(SoftmaxExpCode)
exp_ker = softmax_exp_mod.get_function('softmax_exp')

softmax_mean_mod = SourceModule(SoftmaxMeanCode)
mean_ker = softmax_mean_mod.get_function('softmax_mean')