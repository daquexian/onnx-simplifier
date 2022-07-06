from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from collections import namedtuple
from contextlib import contextmanager
import glob
import os
import shlex
import subprocess
import sys
import platform
from textwrap import dedent
import multiprocessing


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnxsim')
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, '.setuptools-cmake-build')

WINDOWS = (os.name == 'nt')

CMAKE = find_executable('cmake')

install_requires = []
setup_requires = []

################################################################################
# Global variables for controlling the build variant
################################################################################

USE_MSVC_STATIC_RUNTIME = bool(os.getenv('USE_MSVC_STATIC_RUNTIME', '0') == '1')
ONNX_ML = not bool(os.getenv('ONNX_ML') == '0')
ONNX_VERIFY_PROTO3 = bool(os.getenv('ONNX_VERIFY_PROTO3') == '1')
ONNX_NAMESPACE = os.getenv('ONNX_NAMESPACE', 'onnx')
ONNX_BUILD_TESTS = bool(os.getenv('ONNX_BUILD_TESTS') == '1')
ONNX_OPT_USE_SYSTEM_PROTOBUF = bool(os.getenv('ONNX_OPT_USE_SYSTEM_PROTOBUF', '0') == '1')

DEBUG = bool(os.getenv('DEBUG'))
COVERAGE = bool(os.getenv('COVERAGE'))

################################################################################
# Version
################################################################################

try:
    version = subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0'],
                                      cwd=TOP_DIR).decode('ascii').strip()
    if version[0] == 'v':
        version = version[1:]
except (OSError, subprocess.CalledProcessError):
    version = "unknown"

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
    version=version,
    git_version=git_version
)

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


################################################################################
# Customized commands
################################################################################


class ONNXCommand(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class create_version(ONNXCommand):
    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(dedent('''\
            # This file is generated by setup.py. DO NOT EDIT!

            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setupmnm.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """
    user_options = [
        (str('jobs='), str('j'), str('Specifies the number of jobs to use with make'))
    ]

    built = False

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options('build', ('parallel', 'jobs'))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            build_type = 'Release'
            # configure
            cmake_args = [
                CMAKE,
                '-DPython_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
                '-DPython_EXECUTABLE={}'.format(sys.executable),
                # For pybind11
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                '-DBUILD_ONNX_PYTHON=OFF',
                '-DONNXSIM_PYTHON=ON',
                '-DONNXSIM_BUILTIN_ORT=OFF',
                '-DONNX_USE_LITE_PROTO=OFF',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DONNX_NAMESPACE={}'.format(ONNX_NAMESPACE),
                '-DPY_EXT_SUFFIX={}'.format(
                    sysconfig.get_config_var('EXT_SUFFIX') or ''),
                '-DONNX_OPT_USE_SYSTEM_PROTOBUF={}'.format(
                    'ON' if ONNX_OPT_USE_SYSTEM_PROTOBUF else 'OFF'),
            ]
            if COVERAGE:
                cmake_args.append('-DONNX_COVERAGE=ON')
            if COVERAGE or DEBUG:
                # in order to get accurate coverage information, the
                # build needs to turn off optimizations
                build_type = 'Debug'
            cmake_args.append('-DCMAKE_BUILD_TYPE=%s' % build_type)
            if WINDOWS:
                cmake_args.extend([
                    # we need to link with libpython on windows, so
                    # passing python version to window in order to
                    # find python in cmake
                    '-DPY_VERSION={}'.format('{0}.{1}'.format(* \
                                                              sys.version_info[:2])),
                ])
                if USE_MSVC_STATIC_RUNTIME:
                    cmake_args.append('-DONNX_USE_MSVC_STATIC_RUNTIME=ON')
                if platform.architecture()[0] == '64bit':
                    cmake_args.extend(['-A', 'x64', '-T', 'host=x64'])
                else:
                    cmake_args.extend(['-A', 'Win32', '-T', 'host=x86'])
            if ONNX_ML:
                cmake_args.append('-DONNX_ML=1')
            if ONNX_VERIFY_PROTO3:
                cmake_args.append('-DONNX_VERIFY_PROTO3=1')
            if ONNX_BUILD_TESTS:
                cmake_args.append('-DONNX_BUILD_TESTS=ON')
            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            print(f"Run command {cmake_args}")
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, '--build', os.curdir, '--target onnxsim_cpp2py_export']
            if WINDOWS:
                build_args.extend(['--config', build_type])
                build_args.extend(['--', '/maxcpucount:{}'.format(self.jobs)])
            else:
                build_args.extend(['--', '-j', str(self.jobs)])
            print(f"Run command {build_args}")
            subprocess.check_call(build_args)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        return setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('build_py')
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command('cmake_build')
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            if os.name == 'nt':
                debug_lib_dir = os.path.join(lib_path, "Debug")
                release_lib_dir = os.path.join(lib_path, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_path = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_path = release_lib_dir
            src = os.path.join(lib_path, filename)
            dst_dir = os.path.join(os.path.realpath(
                self.build_lib), "onnxsim")
            dst = os.path.join(dst_dir, filename)
            os.makedirs(dst_dir, exist_ok=True)
            self.copy_file(src, dst)


cmdclass = {
    'create_version': create_version,
    'cmake_build': cmake_build,
    'build_ext': build_ext,
    'build_py': build_py,
    'develop': develop,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(
        name=str('onnxsim.onnxsim_cpp2py_export'),
        sources=[])
]

################################################################################
# Packages
################################################################################

# no need to do fancy stuff so far
packages = setuptools.find_packages()

install_requires.extend([
    'onnx',
    'rich',
    'onnxruntime',
])

################################################################################
# Test
################################################################################

setup_requires.append('pytest-runner')

################################################################################
# Final
################################################################################

setuptools.setup(
    name="onnxsim",
    version=VersionInfo.version,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    license='Apache License v2.0',
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    author='ONNX Simplifier Authors',
    author_email='daquexian566@gmail.com',
    url='https://github.com/daquexian/onnx-simplifier',
)
