#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
struct lua_State;

lua_State *luaInit(void);
void luaClose(lua_State *L);

class _Linear {
    lua_State *L;
//    int inputSize;
//    int outputSize;

public:

    _Linear(lua_State *L, int inputSize, int outputSize);
    ~_Linear();
    THFloatTensor *updateOutput(THFloatTensor *input);
    THFloatTensor *getWeight();
    THFloatTensor *getOutput();
};

