// {{header1}}
// {{header2}}

extern "C" {
    #include "lua.h"
    #include "lauxlib.h"
    #include "lualib.h"
}

#ifndef _WIN32
    #include <dlfcn.h>
#endif

#include <iostream>
#include <stdexcept>

#include "luaT.h"
#include "THTensor.h"
#include "THStorage.h"
#include "LuaHelper.h"
#include "nnWrapper.h"

using namespace std;

// adapted from http://stackoverflow.com/questions/12256455/print-stacktrace-from-c-code-with-embedded-lua
static int traceback (lua_State *L) {
//  cout << "traceback()" << endl;
//  if (!lua_isstring(L, -1)) {  /* 'message' not a string? */
//    cout << "message not a string " << endl;
//    return 1;  /* keep it intact */
//  }
  lua_getglobal(L, "debug");
//  if (!lua_istable(L, -1)) {
//    cout << "no debug" << endl;
//    lua_pop(L, 1);
//    return 1;
//  }
  lua_getfield(L, -1, "traceback");
//  if (!lua_isfunction(L, -1)) {
//    cout << "no traceback" << endl;
//    lua_pop(L, 2);
//    return 1;
//  }
  lua_remove(L, -2);

  lua_pushthread(L);
  lua_pushvalue(L, -3);  /* pass error message */
  lua_pushinteger(L, 3);  /* skip this function and traceback */
  lua_call(L, 3, 1);  /* call debug.traceback */
//  cout << "traceback: " << lua_tostring(L, -1) << endl;
  return 1;
}

lua_State *luaInit() {
    #ifndef _WIN32
    #ifdef USE_LUAJIT
        #define LUALIBNAME "libluajit"
    #else
        #define LUALIBNAME "libPyTorchLua"
    #endif
    void *hdl = dlopen(LUALIBNAME ".so", RTLD_NOW | RTLD_GLOBAL);
    if(hdl == 0) {
        hdl = dlopen(LUALIBNAME ".dylib", RTLD_NOW | RTLD_GLOBAL);
    }
    if(hdl == 0) {
        cout << "Failed to load both " LUALIBNAME ".so and " LUALIBNAME ".dylib, fatal" << endl;
        cout << dlerror() << endl;
        throw runtime_error(string("Couldnt load " LUALIBNAME ".so or " LUALIBNAME ".dylib") + dlerror());
    } else {
    }

    #endif

    lua_State *L = luaL_newstate();
    luaL_openlibs(L);

    // see http://stackoverflow.com/questions/12256455/print-stacktrace-from-c-code-with-embedded-lua/16323388#16323388
    lua_pushcfunction(L, traceback); // so, this will always be at stack position 1.  in theory...

    lua_getglobal(L, "require");
    lua_pushstring(L, "torch");
    lua_call(L, 1, 0);
//    lua_setglobal(L, "torch");

    lua_getglobal(L, "require");
    lua_pushstring(L, "nn");
    lua_call(L, 1, 1);
    lua_setglobal(L, "nn");

    return L;
}
//void luaRequire(lua_State *L, const char *libName) {
//    lua_getglobal(L, "require");
//    lua_pushstring(L, libName);
//    lua_call(L, 1, 1);
//    lua_setglobal(L, libName);
//}
void luaClose(lua_State *L) {
    lua_close(L);
}
void collectGarbage(lua_State *L) {
    pushGlobal(L, "collectgarbage");
    lua_call(L, 0, 0);
}

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
int TH{{Real}}Storage_getRefCount(TH{{Real}}Storage *self) {
    return self->refcount;
}
int TH{{Real}}Tensor_getRefCount(TH{{Real}}Tensor *self) {
    return self->refcount;
}
{% endfor %}

long pointerAsInt(void *ptr) {
    return (long)ptr;
}

