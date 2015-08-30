#pragma once

struct lua_State;

void createInstanceStore(lua_State *L, void *key);
void deleteInstanceStore(lua_State *L, void *key);
void pushInstanceValue(lua_State *L, void *instanceKey, const char *name);
void getInstanceValue(lua_State *L, void *instanceKey, const char *name);
void getGlobal(lua_State *L, const char *name1);
void getGlobal(lua_State *L, const char *name1, const char *name2);
void getGlobal(lua_State *L, const char *name1, const char *name2, const char *name3);

