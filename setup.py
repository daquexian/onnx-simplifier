from setuptools import setup, find_packages

setup(
    name='onnx-simplifier',
    version='0.1.5',
    description='Simplify your ONNX model',
    author='daquexian',
    author_email='daquexian566@gmail.com',
    url='https://github.com/daquexian/onnx-simplifier',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
    license='Apache',
    keywords='deep-learning ONNX',
    install_requires=[
        'onnx',
        'onnxruntime >= 0.3.0',
        'protobuf >= 3.7.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.5'
)
