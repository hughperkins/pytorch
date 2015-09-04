#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
class THFloatStorage;
class THLongTensor;
class THLongStorage;
struct lua_State;

lua_State *luaInit(void);
void luaClose(lua_State *L);

long pointerAsInt(void *ptr); // mostly for debugging
void collectGarbage(lua_State *L);

int THFloatStorage_getRefCount(THFloatStorage *self);
int THLongStorage_getRefCount(THLongStorage *self);
int THFloatTensor_getRefCount(THFloatTensor *self);
int THLongTensor_getRefCount(THLongTensor *self);

