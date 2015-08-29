#pragma once

//#include "luaL.h"
//#include "luaT.h"

#include <iostream>
#include <string>

class THFloatTensor;
struct lua_State;

lua_State *luaInit(void);
void luaClose(lua_State *L);

class _Linear {
public:
    int inputSize;
    int outputSize;

    _Linear(lua_State *L, int inputSize, int outputSize) {
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        std::cout << "_Linear()" << std::endl;
    }
    ~_Linear() {
        std::cout << "~_Linear()" << std::endl;
    }
    void updateOutput(THFloatTensor *input) {
        std::cout << "updateOutput..." << std::endl;
    }
    THFloatTensor *getOutput() {
        std::cout << "getOutput..." << std::endl;
        return 0;
    }
};

