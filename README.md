# pytorch
Wrappers to use torch and lua from python

# What is implemented

See [Implemented.md](doc/Implemented.md)

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

* (New!) you can also import your own lua class, and call methods on those :-)

# Requirements

- python should be in PATH
- Cython should be installed, ie if you do `pip freeze`, you should see a list of libraries, and one of those should be `Cython`
- torch should have been activated, ie something like `source ~/torch/install/bin/torch-activate.sh`
- lua51 headers should be installed, ie something like `sudo apt-get install lua5.1 liblua5.1-dev`

# Unit-tests

Run:
```
./build.sh
./run_tests.sh
```

* [test](test) folder, containing test scripts
* [tests_output.txt](test_outputs/tests_output.txt)

# pytorch, example using FloatTensor

Run example script by doing:
```
./build.sh
./run.sh
```

* [test_pytorch.py](test/test_pytorch.py)
* [test_pytorch_output.txt](test_outputs/test_pytorch_output.txt)

# pynn

Run example script by doing:
```
./build.sh
./nn_run.sh
```

* [test_pynn.py](test/test_pynn.py)
* [test_pynn_output.txt](test_outputs/test_pynn_output.txt)

# import your own class, and call methods on it

Create a lua class, like say:

```
require 'torch'

print('foo.lua')

function func()
  print('func()')
end

local Foo = torch.class('Foo')

function Foo:__init()
  print('Foo:__init()')
  self.color = color
end

function Foo:teststuff()
  print('Foo:teststuff()')
end

function Foo:teststuff2(text)
  print('Foo:teststuff2(', text, ')')
end

function Foo:setColor(color)
  print('setColor:', color)
  self.color = color
end

function Foo:printColor()
  print('color:', self.color)
end
```

Create a python script, like say:
```
import PyTorch
import sys
import os
import PyTorchAug

PyTorch.require('foo')

class Foo(PyTorchAug.LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'Foo'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__([name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()


foo = Foo()
foo.teststuff()
foo.teststuff2('hello')
foo.setColor('green')
foo.printColor()
```
Sorry about the magical incandation for the class definition.  But basically it `require`s foo, then
creates a `Foo` object, then calls methods on that object :-)

When we run it we get:
```
foo.lua
Foo:__init()
Foo:teststuff()
Foo:teststuff2(	hello	)
setColor:	green
color:	green
```

# Installation

## Pre-requisites

* Have installed torch, following instructions at [https://github.com/torch/distro](https://github.com/torch/distro)
* Have installed 'nn', ie:
```
luarocks install nn
```
* Have installed python (tested with 2.7 and 3.4)
* Have installed the following python libraries, ie do:
```
pip install numpy
pip install Cython
pip install Jinja2
pip install pytest
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
* simply copy the appropriate declaration into [PyTorch.jinja2.pyx](src/PyTorch.jinja2.pyx), in the blocks that start `cdef extern from "THStorage.h":` or `cdef extern from "THTensor.h":`, as appropriate
  * note that this is a template file.  The generated Cython .pyx file is [PyTorch.pyx](src/PyTorch.pyx)
* and add an appropriate method into {{Real}}Storage class, or {{Real}}Tensor class
* that's it :-)

You can have a look eg at the `narrow` method as an example

Updates:
* the cython class is now called eg `_{{Real}}Tensor` instead of `{{Real}}Tensor`.  Then we create a pure Python class called `{{Real}}Tensor` around this, by inheriting from `_{{Real}}Tensor`, and providing no additional methods or properties.  The pure Python class is monkey-patchable, eg by [PyClTorch](https://github.com/hughperkins/pycltorch)
* the `cdef` properties and methods are now declared in [PyTorch.jinja2.pxd](src/PyTorch.jinja2.pxd).  This means we can call these methods and properties from [PyClTorch](https://github.com/hughperkins/pycltorch)
  * note that [PyTorch.jinja2.pxd](src/PyTorch.jinja2.pxd) is a template file.  The generated Cython .pxd file is [PyTorch.pxd](src/PyTorch.pxd)
* the `THGenerator *` object is now available, at `globalState.generator`, and used by the various random methods, eg `_FloatTensor.bernoulli()`

## pynn

This has been simplified a bunch since before.  We no longer try to wrap C++ classes around the lua, but just directly wrap Python classes around the Lua.  The class methods and attributes are mostly generated dynamically, according to the results of querying hte lua ones.  Mostly all we have to do is create classes with appropriate names, that derive from LuaClass.  We might need to handle inheritance somehow in the future.  At the moment, this is all handled really by PyTorchAug.py.

# Related projects

* [pycltorch](https://github.com/hughperkins/pycltorch) python wrappers for [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn)
* [pycudatorch](https://github.com/hughperkins/pycudatorch) python wrappers for [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn)

# Recent news

24th February:
* added support for passing strings to methods
* added `require`
* created prototype for importing your own classes, and calling methods on those
* works with Python 3 now :-)

12th December:
* created [Implemented.md](doc/Implemented.md) doc
* Added several network layers:
  * Reshape
  * SpatialConvolutionMM
  * SpatialMaxPooling
  * Tanh
  * ReLU

5th September:
* added DoubleTensor
* added ByteTensor
* moved test scripts and output out of readme, provide links instead
* test output linked from readme updated automatically
* added + - * / for tensor/scalar pairs, and + - for tensor pairs

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


