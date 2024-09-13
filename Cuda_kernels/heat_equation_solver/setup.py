from setuptools import setup, Extension, find_packages
import os
import numpy
import subprocess

# Detect if it's in development mode based on the environment variable
development_mode = os.getenv("DEVELOPMENT") == "1"
install_type = "dev_install" if development_mode else "final_install"

# Determine the installation directory based on development or final install
install_dir = os.path.join(os.getcwd(), "build", install_type)

# Add include directories where the headers are located
include_dirs = [
    numpy.get_include(),
    os.path.join(install_dir, "include"),  # C++ headers installed by CMake
    os.path.join(os.getcwd(), "include")   # Local source headers
]

# Library directories where CMake installed shared libraries
library_dirs = [
    os.path.join(install_dir, "lib")  # Directory where libheat_solver.so is located
]

# Extra link arguments for CUDA linking
extra_link_args = [
    f'-L{os.path.join(install_dir, "lib")}',  # Library path
    '-lcudart'  # Link against CUDA runtime library
]

# Build CMake if needed
def build_cmake():
    build_dir = os.path.join(os.getcwd(), "build")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    subprocess.check_call(['cmake', f'-DDEVELOPMENT={"ON" if development_mode else "OFF"}', '..'], cwd=build_dir)
    subprocess.check_call(['make', '-j'], cwd=build_dir)
    subprocess.check_call(['make', 'install'], cwd=build_dir)

# Build CMake if necessary
build_cmake()

# Define the extension module that wraps the C++ and CUDA code
ext_modules = [
    Extension(
        name="heat_solver",
        sources=["src/pycuda_wrapper.cpp"],  # Python wrapper for C++/CUDA code
        include_dirs=include_dirs,  # Include directories (NumPy, C++, etc.)
        library_dirs=library_dirs,  # Library directories for libheat_solver.so
        libraries=["heat_solver"],  # The shared library created by CMake
        extra_compile_args=['-fopenmp'],  # OpenMP support
        extra_link_args=extra_link_args,  # Additional link flags
        language="c++",  # Specify the language (C++)
    )
]

# Setup function to configure the Python package
setup(
    name="heat_solver",
    version="1.0",
    description="A heat equation solver using CUDA and OpenMP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    install_requires=['pycuda', 'numpy'],
    include_package_data=True,
    package_data={
        '': [f'{install_dir}/lib/libheat_solver.so'],  # Include the shared library
    },
    zip_safe=False,  # Ensures that the package can handle dynamic loading
)
