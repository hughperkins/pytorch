#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
class THFloatStorage;
struct lua_State;

lua_State *luaInit(void);
void luaClose(lua_State *L);

long pointerAsInt(void *ptr); // mostly for debugging
void collectGarbage(lua_State *L);
int THFloatStorage_getRefCount(THFloatStorage *self);
int THFloatTensor_getRefCount(THFloatTensor *self);

