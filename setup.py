# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import os
import os.path
import sys
import platform
from setuptools import setup
from setuptools import Extension

home_dir = os.getenv('HOME')
print('home_dir:', home_dir)

cython_present = False
from Cython.Build import cythonize
cython_present = True

jinja_present = False
import jinja2
jinja_present = True

# first generate cython pyx from jinja template...
from jinja2 import Environment, PackageLoader, Template
env = Environment(loader=jinja2.FileSystemLoader('.'))
templateNames = [
    'src/PyTorch.jinja2.pyx', 'src/Storage.jinja2.pyx', 'src/PyTorch.jinja2.pxd', 'src/nnWrapper.jinja2.cpp', 'src/nnWrapper.jinja2.h',
    'test/jinja2.test_pytorch.py', 'src/Storage.jinja2.pxd', 'src/nnWrapper.jinja2.pxd', 'src/lua.jinja2.pxd', 'src/lua.jinja2.pyx']
for templateName in templateNames:
    template = env.get_template(templateName)
    pyx = template.render(
        header='GENERATED FILE, do not edit by hand, ' +
        'Source: ' + templateName,
        header1='GENERATED FILE, do not edit by hand',
        header2='Source: ' + templateName)
    outFilename = templateName.replace('.jinja2', '').replace('jinja2.', '')
    print('outfilename', outFilename)
    isUpdate = True
    if os.path.isfile(outFilename):
        # read existing file, see if anything changed
        f = open(outFilename, 'rb')  # binary, so get linux line endings, even on Windows
        pyx_current = f.read()
        f.close()
        if pyx_current == pyx:
            isUpdate = False
    if isUpdate:
        print(outFilename + ' (changed)')
        f = open(outFilename, 'wb')
        f.write(pyx)
        f.close()

building_dist = False
for arg in sys.argv:
    if arg in ('sdist', 'bdist', 'bdist_egg', 'build_ext'):
        building_dist = True
        break

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
    compile_options.append('/EHsc')
elif osfamily == 'Linux':
    compile_options.append('-std=c++0x')
    compile_options.append('-g')
    if 'DEBUG' in os.environ:
        compile_options.append('-O0')
else:
    pass
    # put other options etc here if necessary

runtime_library_dirs = []
libraries = []
#libraries.append('lua5.1')
#libraries.append('luaT')
#libraries.append('mylib')
libraries.append('TorchLanguageIndependence')
#libraries.append('TH')
library_dirs = []
library_dirs.append('cbuild')
library_dirs.append(home_dir + '/torch/install/lib')

if osfamily == 'Linux':
    runtime_library_dirs = ['.']

if osfamily == 'Windows':
    libraries.append('winmm')

#sources = ["PyTorch.cxx"]
#if cython_present:
#sources = ['src/lua.pyx', 'src/Storage.pyx', "src/PyTorch.pyx"]
ext_modules = [
    Extension("PyTorch",
              sources=['src/PyTorch.pyx'],
              include_dirs=[home_dir + '/torch/install/include/TH', 'thirdparty/lua-5.1.5/src', home_dir + '/torch/install/include'],
              library_dirs=library_dirs,
              libraries=libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++"),
    Extension("Storage",
              sources=['src/Storage.pyx'],
              include_dirs=[home_dir + '/torch/install/include/TH', 'thirdparty/lua-5.1.5/src', home_dir + '/torch/install/include'],
              library_dirs=library_dirs,
              libraries=libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++"),
    Extension("lua",
              sources=['src/lua.pyx'],
              include_dirs=[home_dir + '/torch/install/include/TH', 'thirdparty/lua-5.1.5/src', home_dir + '/torch/install/include'],
              library_dirs=library_dirs,
              libraries=libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++")
]

ext_modules = cythonize(ext_modules)

setup(
    name='PyTorch',
    version='',
    author='Hugh Perkins',
    author_email='hughperkins@gmail.com',
    description=(
        'Python wrappers for torch and nn'),
    license='BSD2',
    url='https://github.com/hughperkins/pytorch',
    long_description='Python wrappers for torch and nn',
    classifiers=[
    ],
    install_requires=['Cython', 'numpy', 'Jinja2'],
    scripts=[],
    ext_modules=ext_modules,
    py_modules=['src/floattensor', 'src/PyTorchAug'],
)

