# Numerical Differentiation

This project implements numerical methods for differentiating mathematical functions. It includes forward, backward, and central differentiation methods of priliminary functions. 

## Project Structure

```
Numerical-Differentiation
├── docs
│   └── Numerical-differention.pdf
├── Makefile
├── README.md
└── src
    ├── docs
    ├── include
    │   ├── BackwardDerivative.h
    │   ├── CentralDerivative.h
    │   ├── ForwardDerivative.h
    │   └── MathFunctions.h
    ├── main.cpp
    ├── src
    │   ├── BackwardDerivative.cpp
    │   ├── CentralDerivative.cpp
    │   ├── ForwardDerivative.cpp
    │   └── MathFunctions.cpp
    └── tests
        └── MathFunctionsTest.cpp
```

## Building the Project

To compile the project, ensure you have `nvcc`, `nsight`, `g++` and `make` installed on your system. Navigate to the project directory and run the following command:

```bash
make all
```


This will compile all source files and link them into an executable named derivative_cuda.

## Profiling with Nsight Systems

Nsight Systems allows you to profile your CUDA applications to understand performance bottlenecks. Follow these steps to profile the application using Nsight Systems and generate a report:


### Run Nsight Systems:
```bash
nsys profile --trace=cuda,nvtx --output=derivative_cuda ./derivative_cuda
```

```

[1/8] [========================100%] derivative_cuda-rep.nsys-rep
[2/8] [========================100%] derivative_cuda-rep.sqlite
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Style                 Range               
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------  ----------------------------------
     56.0          224,515          1  224,515.0  224,515.0   224,515   224,515          0.0  PushPop  Forward Sine Kernel Launch        
      6.6           26,520          1   26,520.0   26,520.0    26,520    26,520          0.0  PushPop  Backward Sine Kernel Launch       
      6.4           25,508          1   25,508.0   25,508.0    25,508    25,508          0.0  PushPop  Central Sine Kernel Launch        
      6.2           25,007          1   25,007.0   25,007.0    25,007    25,007          0.0  PushPop  Backward Exponential Kernel Launch
      6.1           24,547          1   24,547.0   24,547.0    24,547    24,547          0.0  PushPop  Forward Exponential Kernel Launch 
      5.7           22,863          1   22,863.0   22,863.0    22,863    22,863          0.0  PushPop  Central Exponential Kernel Launch 
      4.6           18,455          1   18,455.0   18,455.0    18,455    18,455          0.0  PushPop  Forward Polynomial Kernel Launch  
      4.5           18,124          1   18,124.0   18,124.0    18,124    18,124          0.0  PushPop  Backward Polynomial Kernel Launch 
      3.9           15,479          1   15,479.0   15,479.0    15,479    15,479          0.0  PushPop  Central Polynomial Kernel Launch  

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)        Name     
 --------  ---------------  ---------  -----------  -----------  --------  -----------  ------------  --------------
     44.9      125,106,361      3,072     40,724.7        591.0       440  123,262,661   2,223,918.3  putc          
     36.2      100,761,992         11  9,160,181.1  1,388,083.0     2,825   46,102,446  15,166,938.4  poll          
     17.2       47,816,242        642     74,480.1      8,831.0       480   14,323,653     661,459.6  ioctl         
      0.8        2,273,878         28     81,209.9      8,511.5     6,282    1,489,615     277,813.8  mmap64        
      0.3          897,615     24,576         36.5         31.0        20        5,771          38.2  fwrite        
      0.1          379,126          9     42,125.1     24,937.0     8,195      214,585      65,400.2  sem_timedwait 
      0.1          366,001         52      7,038.5      6,928.0     3,156       12,944       1,639.9  open64        
      0.1          159,313         50      3,186.3      2,540.0     1,553       10,340       1,744.4  fopen         
      0.0          135,896         15      9,059.7      3,226.0     1,614       72,517      17,786.5  mmap          
      0.0          117,412          3     39,137.3     37,691.0    30,908       48,813       9,039.7  pthread_create
      0.0           90,615      3,082         29.4         30.0        20          762          15.4  fflush        
      0.0           61,375         51      1,203.4         30.0        20       59,703       8,355.4  fgets         
      0.0           54,966         44      1,249.2      1,207.0       862        1,754         184.7  fclose        
      0.0           26,398          5      5,279.6      5,430.0     3,877        6,171         860.9  munmap        
      0.0           24,255          6      4,042.5      4,057.5     2,004        6,492       1,568.3  open          
      0.0           19,547         59        331.3        330.0       210          841         102.7  fcntl         
      0.0           17,182          2      8,591.0      8,591.0     7,284        9,898       1,848.4  fread         
      0.0           13,744         13      1,057.2        922.0       511        2,625         590.6  read          
      0.0           13,326          2      6,663.0      6,663.0     4,048        9,278       3,698.2  socket        
      0.0           11,481         10      1,148.1      1,162.0       461        1,954         484.4  write         
      0.0            8,907          1      8,907.0      8,907.0     8,907        8,907           0.0  connect       
      0.0            6,983          1      6,983.0      6,983.0     6,983        6,983           0.0  pipe2         
      0.0            1,916          7        273.7        301.0       221          321          44.4  dup           
      0.0            1,533          1      1,533.0      1,533.0     1,533        1,533           0.0  bind          
      0.0              632          1        632.0        632.0       632          632           0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)  Min (ns)   Max (ns)   StdDev (ns)            Name         
 --------  ---------------  ---------  -----------  --------  --------  ----------  ------------  ----------------------
     98.4       59,962,131         27  2,220,819.7   1,412.0       962  59,539,863  11,455,347.6  cudaMalloc            
      0.6          386,715         27     14,322.8   1,563.0       992      52,779      18,756.5  cudaFree              
      0.6          344,290          9     38,254.4  17,172.0    11,011     215,898      66,676.6  cudaLaunchKernel      
      0.3          189,858         27      7,031.8   7,143.0     3,366      16,441       2,996.6  cudaMemcpy            
      0.1           46,046          9      5,116.2   5,751.0     3,486       6,091       1,118.0  cudaDeviceSynchronize 
      0.0            1,373          1      1,373.0   1,373.0     1,373       1,373           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                     Name                                   
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------------------------------------------
     14.8            4,416          1   4,416.0   4,416.0     4,416     4,416          0.0  forwardSinKernel(const double *, double, double *, double *, int)         
     13.8            4,128          1   4,128.0   4,128.0     4,128     4,128          0.0  forwardExponentialKernel(const double *, double, double *, double *, int) 
     13.5            4,032          1   4,032.0   4,032.0     4,032     4,032          0.0  backwardSinKernel(const double *, double, double *, double *, int)        
     13.5            4,032          1   4,032.0   4,032.0     4,032     4,032          0.0  centralSinKernel(const double *, double, double *, double *, int)         
     13.1            3,904          1   3,904.0   3,904.0     3,904     3,904          0.0  centralExponentialKernel(const double *, double, double *, double *, int) 
     12.9            3,840          1   3,840.0   3,840.0     3,840     3,840          0.0  backwardExponentialKernel(const double *, double, double *, double *, int)
      6.3            1,888          1   1,888.0   1,888.0     1,888     1,888          0.0  centralPolynomialKernel(const double *, double, double *, double *, int)  
      6.1            1,824          1   1,824.0   1,824.0     1,824     1,824          0.0  backwardPolynomialKernel(const double *, double, double *, double *, int) 
      6.0            1,792          1   1,792.0   1,792.0     1,792     1,792          0.0  forwardPolynomialKernel(const double *, double, double *, double *, int)  

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------
     70.2           22,400     18   1,244.4   1,216.0     1,184     1,440         62.9  [CUDA memcpy DtoH]
     29.8            9,503      9   1,055.9   1,024.0     1,023     1,216         62.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
      0.147     18     0.008     0.008     0.008     0.008        0.000  [CUDA memcpy DtoH]
      0.074      9     0.008     0.008     0.008     0.008        0.000  [CUDA memcpy HtoD]

Generated:
    /home/data2/ronak/te/Notes/Numerical-Differentiation/derivative_cuda-rep.nsys-rep
    /home/data2/ronak/te/Notes/Numerical-Differentiation/derivative_cuda-rep.sqlite
```

### Generate NVTX GPU Projection Trace:
```bash
nsys stats --report nvtx_gpu_proj_trace derivative_cuda.nsys-rep
```

```bash
Generating SQLite file derivative_cuda-rep.sqlite from derivative_cuda-rep.nsys-rep
Exporting 32642 events: [==================================================100%]
Processing [derivative_cuda-rep.sqlite] with [/usr/local/cuda-12.2/nsight-systems-2023.2.3/host-linux-x64/reports/nvtx_gpu_proj_trace.py]... 

 ** NVTX GPU Projection Trace (nvtx_gpu_proj_trace):

                Name                 Projected Start (ns)  Projected Duration (ns)  Orig Start (ns)  Orig Duration (ns)   Style     PID      TID    NumGPUOps  Lvl   NumChild  RangeId  ParentId  RangeStack
 ----------------------------------  --------------------  -----------------------  ---------------  ------------------  -------  -------  -------  ---------  ----  --------  -------  --------  ----------
 None                                         282,493,820                1,373,678             None                None  None     101,690  101,690         27  None         0     None  None      None      
 Forward Sine Kernel Launch                   282,734,719                    4,416      282,516,397             224,515  PushPop  101,690  101,690          1     0         0        1  None      :1        
 Forward Exponential Kernel Launch            282,911,297                    4,128      282,892,087              24,547  PushPop  101,690  101,690          1     0         0        2  None      :2        
 Forward Polynomial Kernel Launch             283,047,074                    1,792      283,031,481              18,455  PushPop  101,690  101,690          1     0         0        3  None      :3        
 Backward Sine Kernel Launch                  283,184,163                    4,032      283,162,679              26,520  PushPop  101,690  101,690          1     0         0        4  None      :4        
 Backward Exponential Kernel Launch           283,321,861                    3,840      283,301,802              25,007  PushPop  101,690  101,690          1     0         0        5  None      :5        
 Backward Polynomial Kernel Launch            283,453,222                    1,824      283,437,859              18,124  PushPop  101,690  101,690          1     0         0        6  None      :6        
 Central Sine Kernel Launch                   283,588,967                    4,032      283,568,416              25,508  PushPop  101,690  101,690          1     0         0        7  None      :7        
 Central Exponential Kernel Launch            283,721,929                    3,904      283,703,962              22,863  PushPop  101,690  101,690          1     0         0        8  None      :8        
 Central Polynomial Kernel Launch             283,852,554                    1,888      283,839,959              15,479  PushPop  101,690  101,690          1     0         0        9  None      :9        

```
