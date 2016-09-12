// This is used for diagnosing issues related to linking with libPyTorchLua.so
// Mostly for development/maintenance usage

#include <iostream>
using namespace std;

#include <dlfcn.h>

extern "C" {
  #include "luaT.h"
  #include "lualib.h"
int luaopen_libpaths(lua_State *L);
}

int main(int argc, char *argv[]) {
  void *err = dlopen("libPyTorchLua.so", RTLD_NOW | RTLD_GLOBAL);
  cout << "err " << (long)err << endl;
  cout << "dlerror " << dlerror() << endl << endl;
  err = dlopen("/home/ubuntu/torch/install/lib/lua/5.1/libpaths.so", RTLD_NOW | RTLD_GLOBAL);
  cout << "err " << (long)err << endl;


    lua_State *L = luaL_newstate();
    luaL_openlibs(L);

luaopen_libpaths(L);

    lua_getglobal(L, "require");
    lua_pushstring(L, "torch");
    lua_call(L, 1, 0);

  return 0;
}


