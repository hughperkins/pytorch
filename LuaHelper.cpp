#include "LuaHelper.h"

#include "lua.h"

#include <iostream>
using namespace std;

void createInstanceStore(lua_State *L, void *key) {
    lua_pushlightuserdata(L, key);
    lua_newtable(L);
    lua_settable(L, LUA_REGISTRYINDEX);
}
void deleteInstanceStore(lua_State *L, void *key) {
    lua_pushlightuserdata(L, key);
    lua_pushnil(L);
    lua_settable(L, LUA_REGISTRYINDEX);
}
void pushInstanceValue(lua_State *L, void *instanceKey, const char *name) {
    lua_pushlightuserdata(L, instanceKey);
    lua_gettable(L, LUA_REGISTRYINDEX);
    lua_insert(L, -2);

    lua_pushstring(L, name);
    lua_insert(L, -2);

    lua_settable(L, -3);
    lua_remove(L, -1);
}
void getInstanceValue(lua_State *L, void *instanceKey, const char *name) {
    lua_pushlightuserdata(L, instanceKey);
    lua_gettable(L, LUA_REGISTRYINDEX);

    lua_pushstring(L, name);

    lua_gettable(L, -2);
    lua_remove(L, -2);
}
void getGlobal(lua_State *L, const char *name1) {
    lua_getglobal(L, name1);
}
void getGlobal(lua_State *L, const char *name1, const char *name2) {
    lua_getglobal(L, name1);
    lua_getfield(L, -1, name2);
    lua_remove(L, -2);
}
void getGlobal(lua_State *L, const char *name1, const char *name2, const char *name3) {
    lua_getglobal(L, name1);
    lua_getfield(L, -1, name2);
    lua_remove(L, -2);
    lua_getfield(L, -1, name3);
    lua_remove(L, -2);
}

