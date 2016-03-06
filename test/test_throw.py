import PyTorchHelpers
import sys
import os
import traceback

ThrowsError = PyTorchHelpers.load_lua_class('test/test_throw.lua', 'ThrowsError')
throwsError = ThrowsError()

noException = True
try:
  throwsError.go()
except Exception as e:
  noException = False
  print('caught exception', e)
  traceback.print_exc()
#  e.printstacktrace()
assert(not noException)
