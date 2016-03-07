// GENERATED FILE, do not edit by hand
// Source: src/nnWrapper.jinja2.h

#pragma once

#include <iostream>
#include <string>




struct THLongTensor;
struct THLongStorage;

int THLongStorage_getRefCount(THLongStorage *self);
int THLongTensor_getRefCount(THLongTensor *self);



struct THFloatTensor;
struct THFloatStorage;

int THFloatStorage_getRefCount(THFloatStorage *self);
int THFloatTensor_getRefCount(THFloatTensor *self);



struct THDoubleTensor;
struct THDoubleStorage;

int THDoubleStorage_getRefCount(THDoubleStorage *self);
int THDoubleTensor_getRefCount(THDoubleTensor *self);



struct THByteTensor;
struct THByteStorage;

int THByteStorage_getRefCount(THByteStorage *self);
int THByteTensor_getRefCount(THByteTensor *self);



struct THClTensor;
struct THClStorage;

int THClStorage_getRefCount(THClStorage *self);
int THClTensor_getRefCount(THClTensor *self);


struct lua_State;
lua_State *luaInit(void);
void luaClose(lua_State *L);
//void luaRequire(lua_State *L, const char *libName);

long pointerAsInt(void *ptr); // mostly for debugging
void collectGarbage(lua_State *L);
