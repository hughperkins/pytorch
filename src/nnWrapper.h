// GENERATED FILE, do not edit by hand
// Source: src/nnWrapper.jinja2.h

#pragma once



#include <iostream>
#include <string>


class THFloatTensor;
class THFloatStorage;

int THFloatStorage_getRefCount(THFloatStorage *self);
int THFloatTensor_getRefCount(THFloatTensor *self);

class THByteTensor;
class THByteStorage;

int THByteStorage_getRefCount(THByteStorage *self);
int THByteTensor_getRefCount(THByteTensor *self);

class THLongTensor;
class THLongStorage;

int THLongStorage_getRefCount(THLongStorage *self);
int THLongTensor_getRefCount(THLongTensor *self);

class THDoubleTensor;
class THDoubleStorage;

int THDoubleStorage_getRefCount(THDoubleStorage *self);
int THDoubleTensor_getRefCount(THDoubleTensor *self);


struct lua_State;
lua_State *luaInit(void);
void luaClose(lua_State *L);
//void luaRequire(lua_State *L, const char *libName);

long pointerAsInt(void *ptr); // mostly for debugging
void collectGarbage(lua_State *L);
