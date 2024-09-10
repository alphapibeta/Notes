
#!/bin/bash

# Exit if any command fails
set -e

# Build the Heat Equation Solver
echo "Building the Heat Equation Solver..."
cd heat_equation_solver
mkdir -p build
cd build
cmake ../
make -j$(nproc)
cd ../..

# Build the Max Distance Kernel
echo "Building the Max Distance Kernel..."
cd max-distance
mkdir -p build
cd build
cmake ../
make -j$(nproc)
cd ../..

# Run the X86 Profiler for Heat Equation Solver
echo "Running X86 Profiler for Heat Equation Solver..."
python3 X86_profiler.py --build_dirs ./heat_equation_solver/build --kernels gpu_kernel --exec_name ./heat_solver --kernel_args block-x=32 block-y=32 N=1024 nsteps=800 zx=0 --output_file test_heat_final_x86

# Run the X86 Profiler for Max Distance Kernel
echo "Running X86 Profiler for Max Distance Kernel..."
python3 X86_profiler.py --build_dirs ./max-distance/build --kernels computeDistancesKernel reduceMaxKernel --exec_name ./max_distance --kernel_args block-x=32 block-y=32 N=1024 nsteps=1 --output_file test_distance_final_x86

# Launch the Streamlit App
echo "Launching the Streamlit app..."
streamlit run x86_streamlit_app.py ./test_distance_final_x86.csv
