#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
struct lua_State;

lua_State *luaInit(void);
void luaClose(lua_State *L);

long pointerAsInt(void *ptr); // mostly for debugging

class _Module {
protected:
    lua_State *L;
public:
    THFloatTensor *forward(THFloatTensor *input);
    THFloatTensor *backward(THFloatTensor *input, THFloatTensor *gradoutput);
    void zeroGradParameters();
    void updateParameters(float learningRate);
    THFloatTensor *updateOutput(THFloatTensor *input);
    THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *gradoutput);
    THFloatTensor *getOutput();
    THFloatTensor *getGradInput();
};
class _Linear : public _Module {
public:
    _Linear(lua_State *L, int inputSize, int outputSize);
    ~_Linear();
    THFloatTensor *getWeight();
};
class _LogSoftMax : public _Module {
public:
    _LogSoftMax(lua_State *L);
    ~_LogSoftMax();
};
class _Sequential : public _Module {
public:
    _Sequential(lua_State *L);
    ~_Sequential();
    void add(_Module *module);
};
// ===== Criterions =================
class _Criterion {
protected:
    lua_State *L;
public:
    float forward(THFloatTensor *input, THFloatTensor *target);
    float updateOutput(THFloatTensor *input, THFloatTensor *target);
    THFloatTensor *backward(THFloatTensor *input, THFloatTensor *target);
    THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *target);
    float getOutput();
    THFloatTensor *getGradInput();
};
class _MSECriterion : public _Criterion {
public:
    _MSECriterion(lua_State *L);
    ~_MSECriterion();
};
class _ClassNLLCriterion : public _Criterion {
public:
    _ClassNLLCriterion(lua_State *L);
    ~_ClassNLLCriterion();
};
// ============================
class _Dataset {
    lua_State *L;
public:
    _Dataset(lua_State *L);
    ~_Dataset();
};
// ========= trainers  =====================
class _Trainer {
//    lua_State *L;
public:
//    _Trainer(lua_State *L);
//    ~_Trainer();
};
class _StochasticGradient : public _Trainer {
    lua_State *L;
public:
    _StochasticGradient(lua_State *L, _Module *module, _Criterion *criterion);
    ~_StochasticGradient();
    void train(_Dataset *dataset);
};

