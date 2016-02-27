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



lua_State *luaInit() {
    #ifndef _WIN32
    void *hdl = dlopen("liblua5.1.so", RTLD_NOW | RTLD_GLOBAL);
    if(hdl == 0) {
        hdl = dlopen("liblua.dylib", RTLD_NOW | RTLD_GLOBAL);
    }
    if(hdl == 0) {
        cout << dlerror() << endl;
        throw runtime_error(string("Couldnt load liblua5.1.so or liblua.dylib") + dlerror());
    } else {
////        cout << "loaded lua library" << endl;
    }
    #endif

    lua_State *L = luaL_newstate();
    luaL_openlibs(L);

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
