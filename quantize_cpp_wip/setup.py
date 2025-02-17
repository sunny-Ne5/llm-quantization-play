from setuptools import setup, Extension
from torch.utils.cpp_extension import library_paths, CppExtension, BuildExtension

setup(
    name='quantization_core',
    ext_modules=[
        CppExtension(
            'quantization_core',
            sources=['bindings.cpp'],
            library_dirs=library_paths(),  # PyTorch library paths
            libraries=['torch', 'torch_python', 'torch_cpu'],  # Link against PyTorch libraries
            extra_compile_args=['-std=c++17'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)