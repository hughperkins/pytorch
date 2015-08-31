#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
struct lua_State;

lua_State *luaInit(void);
void luaClose(lua_State *L);

class _Module {
protected:
    lua_State *L;
public:
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
class _Criterion {
protected:
    lua_State *L;
public:
    THFloatTensor *updateOutput(THFloatTensor *input);
    THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *target);
};
class _MSECriterion : public _Criterion {
public:
    _MSECriterion(lua_State *L);
    ~_MSECriterion();
};
class _Dataset {
    lua_State *L;
public:
    _Dataset(lua_State *L);
    ~_Dataset();
};
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

