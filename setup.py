from setuptools import setup, find_packages  # type: ignore

install_requires = [
    'onnx',
    'onnxoptimizer >= 0.2.6',
    'protobuf >= 3.7.0',
    'rich'
]

try:
    import onnxruntime
    has_ort = True
except:
    has_ort = False
    install_requires.append('onnxruntime >= 1.10.0')

setup(
    name='onnx-simplifier',
    # The version will be updated automatically in CI
    version='0.0.0',
    description='Simplify your ONNX model',
    author='daquexian',
    author_email='daquexian566@gmail.com',
    url='https://github.com/daquexian/onnx-simplifier',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
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
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'onnxsim=onnxsim:main',
        ],
    },
)
