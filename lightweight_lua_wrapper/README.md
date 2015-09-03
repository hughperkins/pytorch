# lightweight lua wrapper

This just directly wraps lua directly from Python, with minimal C++ layer, eg no C++ classes.  It's an experiment in progress.  Seems to be working fairly well so far...

Note that Lunatic Python looks quite interesting too, http://labix.org/lunatic-python  I've emailed the author to check with him/her:
- do they want to put it on github?
- possible to change from GPL to BSD2?

To try the lightweight lua wrappers in this directory, run:
```
./build.sh
./run.sh
```

- Needs torch, nn installed in Lua.
- Needs Cython installed in python

