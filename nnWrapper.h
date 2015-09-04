// GENERATED FILE, do not edit by hand

#pragma once



#include <iostream>
#include <string>


class THDoubleTensor;
class THDoubleStorage;

int THDoubleStorage_getRefCount(THDoubleStorage *self);
int THDoubleTensor_getRefCount(THDoubleTensor *self);

class THFloatTensor;
class THFloatStorage;

int THFloatStorage_getRefCount(THFloatStorage *self);
int THFloatTensor_getRefCount(THFloatTensor *self);

class THLongTensor;
class THLongStorage;

int THLongStorage_getRefCount(THLongStorage *self);
int THLongTensor_getRefCount(THLongTensor *self);


struct lua_State;
lua_State *luaInit(void);
void luaClose(lua_State *L);

long pointerAsInt(void *ptr); // mostly for debugging
void collectGarbage(lua_State *L);
