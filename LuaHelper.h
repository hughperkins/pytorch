#pragma once

struct lua_State;
struct THFloatTensor;

void dumpStack(lua_State *L);

void *getGlobal(lua_State *L, const char *name1, const char *name2);

void pushGlobal(lua_State *L, const char *name1);
void pushGlobal(lua_State *L, const char *name1, const char *name2);
void pushGlobal(lua_State *L, const char *name1, const char *name2, const char *name3);

void popAsSelf(lua_State *L, void *instanceKey);
void deleteSelf(lua_State *L, void *instanceKey);
void pushSelf(lua_State *L, void *instanceKey);
void getInstanceField(lua_State *L, void *instanceKey, const char *name);
THFloatTensor *popFloatTensor(lua_State *L);
const char * popString(lua_State *L);
float popFloat(lua_State *L);
void pushFloatTensor(lua_State *L, THFloatTensor *tensor);

void require(lua_State *L, const char *name);

