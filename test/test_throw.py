import PyTorchHelpers
import sys
import os
import traceback

def test_FunctionThrow():
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

def test_initThrow():
  ThrowsErrorOnInit = PyTorchHelpers.load_lua_class('test/test_throw.lua', 'ThrowsErrorOnInit')
  noException = True
  try:
    throwsErrorOnInit = ThrowsErrorOnInit()
  except Exception as e:
    noException = False
    print('caught exception', e)
    traceback.print_exc()
  #  e.printstacktrace()
  assert(not noException)

if __name__ == '__main__':
  test_FunctionThrow()
  test_initThrow()

