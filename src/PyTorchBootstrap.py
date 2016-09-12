from __future__ import print_function
import imp
import os
from os import path
from ctypes import cdll, Structure, c_int, c_double, c_uint

os_family = os.uname()[0]
if os_family == 'Linux':
  for lua_cpath in os.environ['LUA_CPATH'].split(';'):
    expanded_path = lua_cpath.replace('?', 'libPyTorchLuajitLoader')
    if path.isfile(expanded_path):
      try:
        lib = cdll.LoadLibrary(expanded_path)
        print('loaded luajit ok from ', expanded_path)
        break
      except:
        print('failed to load luajit at ' + expanded_path)
else:
  print('info: not linux, so skipping bootstrap')

loadLuajit = lib.loadLuajit
loadLuajit()

