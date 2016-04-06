# Examples overview

There are two ways of using pytorch:
- design the network and create/use tensors entirely on the python side.  In this way, you never actually have
to touch any lua at all.  However, in this mode, you lose out a lot on the power of lua/torch.  It's not
entirely working 100%
- keep the torch/lua stuff in lua classes.  Import these classes into python, and create proxy objects, that
run the underlying objects/methods in lua.  In this way, you can use all the power of standard lua/torch,
but call into it easily from your python application, pass in numpy tensors, and so on

For examples of expressing the entire network from within python, ie the first way, you can see for example:
- ../test/test_pytorch.py
- ../test/test_pynn.py

For examples of keeping the lua/torch bits in lua files/classes, and running these from python,
please see for example:
- ../simpleexample/pybit.py
- ./luamodel

