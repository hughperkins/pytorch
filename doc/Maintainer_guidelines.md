# Maintainer guidelines

## How to add new methods

### pytorch

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

### pynn

This has been simplified a bunch since before.  We no longer try to wrap C++ classes around the lua, but just directly wrap Python classes around the Lua.  The class methods and attributes are mostly generated dynamically, according to the results of querying hte lua ones.  Mostly all we have to do is create classes with appropriate names, that derive from LuaClass.  We might need to handle inheritance somehow in the future.  At the moment, this is all handled really by PyTorchAug.py.


