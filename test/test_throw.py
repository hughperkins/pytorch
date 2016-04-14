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
    print('caught successfully raised exception', e)
    traceback.print_exc()
  #  e.printstacktrace()
  assert(not noException)
  print('Note that it\'s normal this throws an exception.  Its a test of exception throwing :-)')

def test_initThrow():
  ThrowsErrorOnInit = PyTorchHelpers.load_lua_class('test/test_throw.lua', 'ThrowsErrorOnInit')
  noException = True
  try:
    throwsErrorOnInit = ThrowsErrorOnInit()
  except Exception as e:
    noException = False
    print('caught successfully raised exception', e)
    traceback.print_exc()
  #  e.printstacktrace()
  assert(not noException)
  print('Note that it\'s normal this throws an exception.  Its a test of exception throwing :-)')

def test_subthrow():
  """
  check that we get the full stack trace, not just the point of failure
  """
  ThrowsError = PyTorchHelpers.load_lua_class('test/test_throw.lua', 'ThrowsError')
  throwsError = ThrowsError()
  try:
    throwsError.insub_anteater()  
  except Exception as e:
    print('error', e)
    assert 'test_throw.lua:18' in str(e)

if __name__ == '__main__':
  test_FunctionThrow()
  test_initThrow()
  test_subthrow()

