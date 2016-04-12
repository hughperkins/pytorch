// GENERATED FILE, do not edit by hand
// Source: src/nnWrapper.jinja2.cpp

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

// from http://stackoverflow.com/questions/12256455/print-stacktrace-from-c-code-with-embedded-lua
static int traceback (lua_State *L) {
  if (!lua_isstring(L, 1))  /* 'message' not a string? */
    return 1;  /* keep it intact */
  lua_getfield(L, LUA_GLOBALSINDEX, "debug");
  if (!lua_istable(L, -1)) {
    lua_pop(L, 1);
    return 1;
  }
  lua_getfield(L, -1, "traceback");
  if (!lua_isfunction(L, -1)) {
    lua_pop(L, 2);
    return 1;
  }
  lua_pushvalue(L, 1);  /* pass error message */
  lua_pushinteger(L, 2);  /* skip this function and traceback */
  lua_call(L, 2, 1);  /* call debug.traceback */
  fprintf(stderr, "%s\n", lua_tostring(L, -1));
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
    lua_pushcfunction(L, traceback);

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




int THLongStorage_getRefCount(THLongStorage *self) {
    return self->refcount;
}
int THLongTensor_getRefCount(THLongTensor *self) {
    return self->refcount;
}



int THFloatStorage_getRefCount(THFloatStorage *self) {
    return self->refcount;
}
int THFloatTensor_getRefCount(THFloatTensor *self) {
    return self->refcount;
}



int THDoubleStorage_getRefCount(THDoubleStorage *self) {
    return self->refcount;
}
int THDoubleTensor_getRefCount(THDoubleTensor *self) {
    return self->refcount;
}



int THByteStorage_getRefCount(THByteStorage *self) {
    return self->refcount;
}
int THByteTensor_getRefCount(THByteTensor *self) {
    return self->refcount;
}


long pointerAsInt(void *ptr) {
    return (long)ptr;
}
