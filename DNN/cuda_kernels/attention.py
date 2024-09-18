from pycuda.compiler import SourceModule

AttentionKernelCode = '''
#include <float.h>

__global__ void compute_attention_scores(float *Q, float *K, float *scores, int batch_size, int seq_length, int d_k)
{
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && row < seq_length && col < seq_length)
    {
        float score = 0.0f;
        for (int i = 0; i < d_k; ++i)
        {
            score += Q[batch_idx * seq_length * d_k + row * d_k + i] * K[batch_idx * seq_length * d_k + col * d_k + i];
        }
        score /= sqrtf((float)d_k);
        scores[batch_idx * seq_length * seq_length + row * seq_length + col] = score;
    }
}

__global__ void apply_softmax(float *scores, int batch_size, int seq_length)
{
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && row < seq_length)
    {
        float max_val = -FLT_MAX;
        for (int i = 0; i < seq_length; ++i)
        {
            float val = scores[batch_idx * seq_length * seq_length + row * seq_length + i];
            if (val > max_val)
                max_val = val;
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < seq_length; ++i)
        {
            float val = expf(scores[batch_idx * seq_length * seq_length + row * seq_length + i] - max_val);
            scores[batch_idx * seq_length * seq_length + row * seq_length + i] = val;
            sum_exp += val;
        }

        for (int i = 0; i < seq_length; ++i)
        {
            scores[batch_idx * seq_length * seq_length + row * seq_length + i] /= sum_exp;
        }
    }
}

__global__ void compute_attention_output(float *scores, float *V, float *output, int batch_size, int seq_length, int d_v)
{
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && row < seq_length && col < d_v)
    {
        float val = 0.0f;
        for (int i = 0; i < seq_length; ++i)
        {
            val += scores[batch_idx * seq_length * seq_length + row * seq_length + i] * V[batch_idx * seq_length * d_v + i * d_v + col];
        }
        output[batch_idx * seq_length * d_v + row * d_v + col] = val;
    }
}
'''

attention_mod = SourceModule(AttentionKernelCode)
compute_attention_scores_ker = attention_mod.get_function('compute_attention_scores')
apply_softmax_ker = attention_mod.get_function('apply_softmax')
compute_attention_output_ker = attention_mod.get_function('compute_attention_output')