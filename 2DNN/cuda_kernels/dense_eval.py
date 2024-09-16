from pycuda.compiler import SourceModule

DenseEvalCode = '''
#define _RELU(x) ( ((x) > 0.0f) ? (x) : 0.0f )
#define _SIGMOID(x)  ( 1.0f / (1.0f + expf(-(x)) ))

__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float * w, float * b, \
                           float * x, float *y, int batch_size, int w_t, int b_t, float delta)
{
     int i = blockDim.x*blockIdx.x + threadIdx.x;

     if (i < num_outputs)
     {
         for(int k=0; k < batch_size; k++)
         {
              double temp = 0.0f;

              for (int j = 0; j < num_inputs; j++)
              {
                  temp += ((double) w[ (num_inputs) * i + j ] ) * ( (double) x[k * num_inputs + j]);
              }

              temp += (double) b[i];

              y[k * num_outputs + i] = (float) temp;
         }

        if( w_t >= 0 && i == (w_t / num_inputs))
        {
              int j = w_t % num_inputs;

              for(int k=0; k < batch_size; k++)
                  y[k*num_outputs + i] += delta*x[k*num_inputs+j];
        }

        if( b_t >= 0 && i == b_t )
        {
              for(int k=0; k < batch_size; k++)
                  y[k*num_outputs + i] += delta;
        }

        if(relu > 0 || sigmoid > 0)
             for(int k=0; k < batch_size; k++)
             {
                  float temp = y[k * num_outputs + i];

                  if (relu > 0)
                      temp = _RELU(temp);

                  if (sigmoid > 0)
                      temp = _SIGMOID(temp);

                  y[k * num_outputs + i] = temp;
             }
    }
    return;
}
'''

dense_eval_mod = SourceModule(DenseEvalCode)
eval_ker = dense_eval_mod.get_function('dense_eval')