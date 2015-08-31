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
    popAsSelf(L, this);

    getGlobal(L, "nn", "Linear", "float");
    pushSelf(L, this);
    lua_call(L, 1, 0);
}
_Linear::~_Linear() {
    deleteSelf(L, this);
}
THFloatTensor *_Module::updateOutput(THFloatTensor *input) {
    getInstanceField(L, this, "updateOutput");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    lua_call(L, 2, 1);
    return popFloatTensor(L);
}
THFloatTensor *_Module::updateGradInput(THFloatTensor *input, THFloatTensor *gradOutput) {
    getInstanceField(L, this, "updateGradInput");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, gradOutput);
    lua_call(L, 3, 1);
    return popFloatTensor(L);
}
THFloatTensor *_Module::getOutput() {
    getInstanceField(L, this, "output");
    return popFloatTensor(L);
}
THFloatTensor *_Module::getGradInput() {
    getInstanceField(L, this, "gradInput");
    return popFloatTensor(L);
}
THFloatTensor *_Linear::getWeight() {
    getInstanceField(L, this, "weight");
    return popFloatTensor(L);
}

_MSECriterion::_MSECriterion(lua_State *L) {
    this->L = L;
    getGlobal(L, "nn", "MSECriterion");
    lua_call(L, 0, 1);
    pushSelf(L, this);
}
_MSECriterion::~_MSECriterion() {
    deleteSelf(L, this);
}
THFloatTensor *_Criterion::updateOutput(THFloatTensor *input) {
    getInstanceField(L, this, "updateOutput");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    lua_call(L, 2, 1);
    return popFloatTensor(L);
}
THFloatTensor *_Criterion::updateGradInput(THFloatTensor *input, THFloatTensor *target) {
    getInstanceField(L, this, "updateGradInput");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, target);
    lua_call(L, 3, 1);
    return popFloatTensor(L);
}
_StochasticGradient::_StochasticGradient(lua_State *L, _Module *module, _Criterion *criterion) {
    this->L = L;
    getGlobal(L, "nn", "MSECriterion");
    pushSelf(L, module);
    pushSelf(L, criterion);
    lua_call(L, 2, 1);
    pushSelf(L, this);
}
_StochasticGradient::~_StochasticGradient() {
    deleteSelf(L, this);
}
void _StochasticGradient::train(_Dataset *dataset) {
}

