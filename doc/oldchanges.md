# Old changes

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

