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
    std::cout << "_Linear()" << std::endl;

    getGlobal(L, "nn", "Linear");
    lua_pushinteger(L, inputSize);
    lua_pushinteger(L, outputSize);
    lua_call(L, 2, 1);
    pushSelf(L, this);

    getGlobal(L, "nn", "Linear", "float");
    getSelf(L, this);
    lua_call(L, 1, 0);

    std::cout << "_Linear() finished" << std::endl;
}
_Linear::~_Linear() {
    std::cout << "~_Linear()" << std::endl;
    deleteSelf(L, this);
}
void _Linear::updateOutput(THFloatTensor *input) {
    std::cout << "updateOutput..." << std::endl;
    getInstanceField(L, this, "updateOutput");
    getSelf(L, this);
    luaT_pushudata(L, input, "torch.FloatTensor");
    lua_call(L, 2, 0);
    std::cout << " ... updateOutput finished" << std::endl;
}
THFloatTensor *_Linear::getWeight() {
    std::cout << "getWeight..." << std::endl;
    return 0;
}
THFloatTensor *_Linear::getOutput() {
    std::cout << "getOutput..." << std::endl;
    getInstanceField(L, this, "output");
    THFloatTensor *tensor = (THFloatTensor *)(*(void **)lua_touserdata(L, -1));
    lua_remove(L, -1);
    cout << "numdims " << THFloatTensor_nDimension(tensor) << endl;
    cout << THFloatTensor_size(tensor, 0) << " " << THFloatTensor_size(tensor, 1) << endl;
    return tensor;
}

