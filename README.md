# pytorch
Wrappers to use torch and lua from python

# What is python?

- create torch tensors, call operations on those, add them together, multiply them, and so on
- create `nn` network modules, pass tensors into those, get the output, and so on
- create your own lua class, call methods on that, pass in tensors
- wrap numpy tensors in torch tensors
  
More info: [Implemented.md](doc/Implemented.md)

# Examples

## pytorch, using FloatTensor

Run example script by doing:
```
source ~/torch/install/bin/torch-activate
cd pytorch
./run.sh
```

* Script is here: [test_pytorch.py](test/test_pytorch.py)
* Output: [test_pytorch_output.txt](test_outputs/test_pytorch_output.txt)

## pynn

Run example script by doing:
```
source ~/torch/install/bin/torch-activate
cd pytorch
./nn_run.sh
```

* Script is here: [test_pynn.py](test/test_pynn.py)
* Script output: [test_pynn_output.txt](test_outputs/test_pynn_output.txt)

## import your own lua class, call methods on it

- Create a lua class, like say [luabit.lua](simpleexample/luabit.lua)
  - it contains some methods, that we will call from Python
- Create a python script, like say [pybit.py](simpleexample/pybit.py)
  - it `require`s foo, then
  - creates a `Foo` object, then
  - calls methods on that object

Run like:
```
source ~/torch/install/bin/torch-activate
cd simpleexample
python pybit.py
```

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
* Have installed 'nn' torch module:
```
luarocks install nn
```
* Have installed python (tested with 2.7 and 3.4)
* Have installed the following python libraries:
```
pip install numpy
pip install pytest
pip install python-mnist  # used for unit tests
```
- lua51 headers should be installed, ie something like `sudo apt-get install lua5.1 liblua5.1-dev`

## Procedure

Run:
```
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
source ~/torch/install/bin/torch-activate
./build.sh
```

# Unit-tests

Run:
```
source ~/torch/install/bin/torch-activate
cd pytorch
./run_tests.sh
```

* The test scripts: [test](test)
* The test output: [tests_output.txt](test_outputs/tests_output.txt)

# Python 2 vs Python 3?

- pytorch is developed and maintained on python 3
- you should be able to use it with python 2, as long as you include the following at the top of your scripts:
```
from __future__ import print_function, division
```

# Maintainer guidelines

[Maintainer guidelines](doc/Maintainer_guidelines.md)

# Versioning

[semantic versioning](http://semver.org/)

# Related projects

* [pycltorch](https://github.com/hughperkins/pycltorch) python wrappers for [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn)
* [pycudatorch](https://github.com/hughperkins/pycudatorch) python wrappers for [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn)

# Recent news

6 March:
* all classes should be usable from `nn` now, without needing to explicitly register inside `pytorch`
  * you need to upgrade to `v3.0.0` to enable this, which is a breaking change, since the `nn` classes are now in `PyTorchAug.nn`, instead of directly
in `PyTorchAug`

5 March:
* added `PyTorchHelpers.load_lua_class(lua_filename, lua_classname)` to easily import a lua class from a lua file
* can pass parameters to lua class constructors, from python
* can pass tables to lua functions, from python (pass in as python dictionaries, become lua tables)
* can return tables from lua functions, to python (returned as python dictionaries)

2 March:
* removed requirements on Cython, Jinja2 for installation

28th Februrary:
* builds ok on Mac OS X now :-)  See https://travis-ci.org/hughperkins/pytorch/builds/112292866

26th February:
* modified `/` to be the div operation for float and double tensors, and `//` for int-type tensors, such as
byte, long, int
* since the div change is incompatible with 1.0.0 div operators, jumping radically from `1.0.0` to `2.0.0-SNAPSHOT` ...
* added dependency on `numpy`
* added `.asNumpyTensor()` to convert a torch tensor to a numpy tensor

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


