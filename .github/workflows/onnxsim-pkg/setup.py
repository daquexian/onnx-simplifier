from setuptools import setup, find_packages  # type: ignore
import os

install_requires = [
    'onnxsim-no-ort==0.0.0',
]

try:
    import onnxruntime
except:
    install_requires.append('onnxruntime >= 1.11.1')

# onnxsim or onnx-simplifier
pkg_name = os.getenv("ONNXSIM_PKG_NAME")
assert pkg_name is not None

setup(
    name=pkg_name,
    # The version will be updated automatically in CI
    version='0.0.0',
    description='Simplify your ONNX model',
    author='daquexian',
    author_email='daquexian566@gmail.com',
    url='https://github.com/daquexian/onnx-simplifier',
    license='Apache',
    keywords='deep-learning ONNX',
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.6',
)

