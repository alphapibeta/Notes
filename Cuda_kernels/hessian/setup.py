import os
import subprocess
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Custom CMake Build Extension class to handle building the C++/CUDA extension
class CMakeBuildExtension(build_ext):
    def build_extensions(self):
        # Get the directory where the shared libraries will be placed
        ext_dir = pathlib.Path(self.get_ext_fullpath(self.extensions[0].name)).parent.absolute()
        build_dir = pathlib.Path('build').absolute()

        if not build_dir.exists():
            os.makedirs(build_dir)

        # Run CMake commands to configure and build the project
        subprocess.check_call(
            ['cmake', '..', f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}', '-DCMAKE_BUILD_TYPE=Release'],
            cwd=build_dir
        )
        subprocess.check_call(['make', '-j4'], cwd=build_dir)

# Define the CMake extension class for the CPU and GPU bindings
class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

# Setup configuration
setup(
    name='hessian_inversion',
    version='0.1',
    author='Ronak Haresh Chhatbar',
    description='Python bindings for Hessian inversion using CPU and GPU',
    long_description=open('README.md').read(),
    packages=['hessian_inversion'],
    ext_modules=[CMakeExtension('hessian_inversion_cpu_py'), CMakeExtension('hessian_inversion_gpu_py')],
    cmdclass={
        'build_ext': CMakeBuildExtension,
    },
    zip_safe=False,
)
