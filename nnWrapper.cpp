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

int THFloatStorage_getRefCount(THFloatStorage *self) {
    return self->refcount;
}
int THFloatTensor_getRefCount(THFloatTensor *self) {
    return self->refcount;
}
long pointerAsInt(void *ptr) {
    return (long)ptr;
}

THFloatTensor *_Module::forward(THFloatTensor *input) {
    getInstanceField(L, this, "forward");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    lua_call(L, 2, 1);
    return popFloatTensor(L);
}
THFloatTensor *_Module::backward(THFloatTensor *input, THFloatTensor *gradOutput) {
    getInstanceField(L, this, "backward");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, gradOutput);
    lua_call(L, 3, 1);
    return popFloatTensor(L);
}
void _Module::zeroGradParameters() {
    getInstanceField(L, this, "zeroGradParameters");
    pushSelf(L, this);
    lua_call(L, 1, 0);
}
void _Module::updateParameters(float learningRate) {
    getInstanceField(L, this, "updateParameters");
    pushSelf(L, this);
    lua_pushnumber(L, learningRate);
    lua_call(L, 2, 0);
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
THFloatTensor *_Linear::getWeight() {
    getInstanceField(L, this, "weight");
    return popFloatTensor(L);
}
_LogSoftMax::_LogSoftMax(lua_State *L) {
    this->L = L;

    getGlobal(L, "nn", "LogSoftMax");
    lua_call(L, 0, 1);
    popAsSelf(L, this);

    getGlobal(L, "nn", "LogSoftMax", "float");
    pushSelf(L, this);
    lua_call(L, 1, 0);
}
_LogSoftMax::~_LogSoftMax() {
    deleteSelf(L, this);
}
_Sequential::_Sequential(lua_State *L) {
    this->L = L;

    getGlobal(L, "nn", "Sequential");
    lua_call(L, 0, 1);
    popAsSelf(L, this);

    getGlobal(L, "nn", "Sequential", "float");
    pushSelf(L, this);
    lua_call(L, 1, 0);
}
_Sequential::~_Sequential() {
    deleteSelf(L, this);
}
void _Sequential::add(_Module *module) {
    getGlobal(L, "nn", "Sequential", "add");
    pushSelf(L, this);
    pushSelf(L, module);
    lua_call(L, 2, 0);
}
// ======== Criterions ==========================
float _Criterion::getOutput() {
    getInstanceField(L, this, "output");
    return popFloat(L);
}
THFloatTensor *_Criterion::getGradInput() {
    getInstanceField(L, this, "gradInput");
    return popFloatTensor(L);
}
float _Criterion::forward(THFloatTensor *input, THFloatTensor *target) {
    getInstanceField(L, this, "forward");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, target);
    lua_call(L, 3, 1);
    return popFloat(L);
}
THFloatTensor *_Criterion::backward(THFloatTensor *input, THFloatTensor *target) {
    getInstanceField(L, this, "backward");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, target);
    lua_call(L, 3, 1);
    return popFloatTensor(L);
}
float _Criterion::updateOutput(THFloatTensor *input, THFloatTensor *target) {
    getInstanceField(L, this, "updateOutput");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, target);
    lua_call(L, 3, 1);
    return popFloat(L);
}
THFloatTensor *_Criterion::updateGradInput(THFloatTensor *input, THFloatTensor *target) {
    getInstanceField(L, this, "updateGradInput");
    pushSelf(L, this);
    pushFloatTensor(L, input);
    pushFloatTensor(L, target);
    lua_call(L, 3, 1);
    return popFloatTensor(L);
}
_MSECriterion::_MSECriterion(lua_State *L) {
    this->L = L;
    getGlobal(L, "nn", "MSECriterion");
    lua_call(L, 0, 1);
    popAsSelf(L, this);

    getGlobal(L, "nn", "MSECriterion", "float");
    pushSelf(L, this);
    lua_call(L, 1, 0);
}
_MSECriterion::~_MSECriterion() {
    deleteSelf(L, this);
}
_ClassNLLCriterion::_ClassNLLCriterion(lua_State *L) {
    this->L = L;
    getGlobal(L, "nn", "ClassNLLCriterion");
    lua_call(L, 0, 1);
    popAsSelf(L, this);

    getGlobal(L, "nn", "ClassNLLCriterion", "float");
    pushSelf(L, this);
    lua_call(L, 1, 0);

    getGlobal(L, "torch", "type");
    pushSelf(L, this);
    lua_call(L, 1, 1);
    cout << "nnWrapper.cpp ClassNLLCriterion::_ClassNLLCriterion type " << popString(L) << endl;
}
_ClassNLLCriterion::~_ClassNLLCriterion() {
    deleteSelf(L, this);
}
//=============trainers=================
_StochasticGradient::_StochasticGradient(lua_State *L, _Module *module, _Criterion *criterion) {
    this->L = L;
    getGlobal(L, "nn", "StochasticGradient");
    // hmmm, is this missing self?
    pushSelf(L, module);
    pushSelf(L, criterion);
    // would need to change this from 2 to 3 too:
    lua_call(L, 2, 1);
    popAsSelf(L, this);

    getGlobal(L, "nn", "StochasticGradient", "float");
    pushSelf(L, this);
    lua_call(L, 1, 0);
}
_StochasticGradient::~_StochasticGradient() {
    deleteSelf(L, this);
}
void _StochasticGradient::train(_Dataset *dataset) {
}

