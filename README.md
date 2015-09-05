# pytorch
POC for wrapping torch in python

# Examples

Examples of what is possible currently:
* pytorch
* pynn

Types supported currently:
* FloatTensor
* DoubleTensor
* LongTensor
* ByteTensor

(fairly easy to add others, since templated)

# Unit-tests

Run:
```
./build.sh
./run_tests.sh
```

* [test](test) folder, containing test scripts
* [tests_output.txt](tests_output.txt)

# pytorch, example using FloatTensor

Run example script by doing:
```
./build.sh
./run.sh
```

* [test_pytorch.py](test_pytorch.py)
* [test_pytorch_output.txt](test_pytorch_output.txt)

# pynn

Run example script by doing:
```
./build.sh
./nn_run.sh
```

* [test_pynn.py](test_pynn.py)
* [test_pynn_output.txt](test_pynn_output.txt)

# Installation

## Pre-requisites

* Have installed torch, following instructions at [https://github.com/torch/distro](https://github.com/torch/distro)
* Have installed 'nn', ie:
```
luarocks install nn
```
* Have installed python (tested with 2.7, for now, no theoretical reason why wont work on 3.4 too)
* Have installed the following python libraries, ie do:
```
pip install numpy
pip install Cython
pip install Jinja2
```

## Procedure

```
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
./build.sh
```

# How to add new methods

## pytorch

* the C library methods are defined in the torch library in torch7 repo, in two files:
  * lib/TH/generic/THTensor.h
  * lib/TH/generic/THStorage.h
* simply copy the appropriate declaration into [PyTorch.jinja2.pyx](PyTorch.jinja2.pyx), in the blocks that start `cdef extern from "THStorage.h":` or `cdef extern from "THTensor.h":`, as appropriate
  * note that this is a template file.  The generated Cython .pyx file is [PyTorch.pyx](PyTorch.pyx)
* and add an appropriate method into {{Real}}Storage class, or {{Real}}Tensor class
* that's it :-)

You can have a look eg at the `narrow` method as an example

Updates:
* the cython class is now called eg `_{{Real}}Tensor` instead of `{{Real}}Tensor`.  Then we create a pure Python class called `{{Real}}Tensor` around this, by inheriting from `_{{Real}}Tensor`, and providing no additional methods or properties.  The pure Python class is monkey-patchable, eg by [PyClTorch](https://github.com/hughperkins/pycltorch)
* the `cdef` properties and methods are now declared in [PyTorch.jinja2.pxd](PyTorch.jinja2.pxd).  This means we can call these methods and properties from [PyClTorch](https://github.com/hughperkins/pycltorch)
  * note that [PyTorch.jinja2.pxd](PyTorch.jinja2.pxd) is a template file.  The generated Cython .pxd file is [PyTorch.pxd](PyTorch.pxd)
* the `THGenerator *` object is now available, at `globalState.generator`, and used by the various random methods, eg `_FloatTensor.bernoulli()`

## pynn

This has been simplified a bunch since before.  We no longer try to wrap C++ classes around the lua, but just directly wrap Python classes around the Lua.  The class methods and attributes are mostly generated dynamically, according to the results of querying hte lua ones.  Mostly all we have to do is create classes with appropriate names, that derive from LuaClass.  We might need to handle inheritance somehow in the future.  At the moment, this is all handled really by PyTorchAug.py.

# Recent news
5th September:
* added DoubleTensor
* added ByteTensor
* moved test scripts and output out of readme, provide links instead
* test output linked from readme updated automatically

4th September:
* added LongTensor
* `size()` now returns a LongTensor, rather than a FloatTensor
* under the covers:
  * started to use Jinja2 as a templating language, means easy to support other types

3rd September:
* modified Lua wrapper approach, so directly uses dynamic Python to wrap the Lua classes

2nd September:
* monkey-patchable, so can start work on [PyClTorch](https://github.com/hughperkins/pycltorch)

29th August:
* first created :-)


