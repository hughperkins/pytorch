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
#include "LuaHelper.h"
#include "nnWrapper.h"

using namespace std;

lua_State *luaInit() {
    #ifndef _WIN32
    void *hdl = dlopen("liblua5.1.so", RTLD_NOW | RTLD_GLOBAL);
    if(hdl == 0) {
        cout << dlerror() << endl;
        throw runtime_error(string("Couldnt load liblua5.1.so ") + dlerror());
    } else {
        cout << "loaded lua library" << endl;
    }
    #endif

    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    lua_getglobal(L, "require");
    lua_pushstring(L, "nn");
    lua_call(L, 1, 1);
    lua_setglobal(L, "nn");

    return L;
}
void luaClose(lua_State *L) {
    lua_close(L);
}

_Linear::_Linear(lua_State *L, int inputSize, int outputSize) {
    this->L = L;

    getGlobal(L, "nn", "Linear");
    lua_pushinteger(L, inputSize);
    lua_pushinteger(L, outputSize);
    lua_call(L, 2, 1);
    pushSelf(L, this);

    getGlobal(L, "nn", "Linear", "float");
    getSelf(L, this);
    lua_call(L, 1, 0);
}
_Linear::~_Linear() {
    deleteSelf(L, this);
}
THFloatTensor *_Linear::updateOutput(THFloatTensor *input) {
    getInstanceField(L, this, "updateOutput");
    getSelf(L, this);
    luaT_pushudata(L, input, "torch.FloatTensor");
    lua_call(L, 2, 1);
    THFloatTensor *tensor = (THFloatTensor *)(*(void **)lua_touserdata(L, -1));
    lua_remove(L, -1);
    return tensor;
}
THFloatTensor *_Linear::getWeight() {
    getInstanceField(L, this, "weight");
    THFloatTensor *tensor = (THFloatTensor *)(*(void **)lua_touserdata(L, -1));
    lua_remove(L, -1);
    return tensor;
}
THFloatTensor *_Linear::getOutput() {
    getInstanceField(L, this, "output");
    THFloatTensor *tensor = (THFloatTensor *)(*(void **)lua_touserdata(L, -1));
    lua_remove(L, -1);
    return tensor;
}

