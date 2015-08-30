#include "LuaHelper.h"

#include "lua.h"
#include "luaT.h"
#include "THTensor.h"

#include <iostream>
using namespace std;

void dumpStack(lua_State *L) {
    cout << "luatop " << lua_gettop(L) << endl;
    if(lua_gettop(L) >= 1 ) cout << "-1 type " << lua_typename(L, lua_type(L, -1)) << endl;
    if(lua_gettop(L) >= 2 ) cout << "-2 type " << lua_typename(L, lua_type(L, -2)) << endl;
    if(lua_gettop(L) >= 3 ) cout << "-3 type " << lua_typename(L, lua_type(L, -3)) << endl;
}

void popAsSelf(lua_State *L, void *instanceKey) {
    lua_pushlightuserdata(L, instanceKey);
    lua_insert(L, -2);
    lua_settable(L, LUA_REGISTRYINDEX);
}
void deleteSelf(lua_State *L, void *instanceKey) {
    lua_pushlightuserdata(L, instanceKey);
    lua_pushnil(L);
    lua_settable(L, LUA_REGISTRYINDEX);
}
void pushSelf(lua_State *L, void *instanceKey) {
    lua_pushlightuserdata(L, instanceKey);
    lua_gettable(L, LUA_REGISTRYINDEX);
}
void getInstanceField(lua_State *L, void *instanceKey, const char *name) {
    lua_pushlightuserdata(L, instanceKey);
    lua_gettable(L, LUA_REGISTRYINDEX);
    lua_getfield(L, -1, name);
    lua_remove(L, -2);
}
THFloatTensor *popFloatTensor(lua_State *L) {
    void **pTensor = (void **)lua_touserdata(L, -1);
    THFloatTensor *tensor = (THFloatTensor *)(*pTensor);
    lua_remove(L, -1);
    return tensor;
}
void pushFloatTensor(lua_State *L, THFloatTensor *tensor) {
    luaT_pushudata(L, tensor, "torch.FloatTensor");
}

//void getInstanceField(lua_State *L, void *instanceKey, const char *name) {
//    getInstanceValue(L, instanceKey);
//    lua_getfield(L, -1, name);
//    lua_remove(L, -2);
//}
//void createInstanceStore(lua_State *L, void *key) {
//    lua_pushlightuserdata(L, key);
//    lua_newtable(L);
//    lua_settable(L, LUA_REGISTRYINDEX);
//}
//void deleteInstanceStore(lua_State *L, void *key) {
//    lua_pushlightuserdata(L, key);
//    lua_pushnil(L);
//    lua_settable(L, LUA_REGISTRYINDEX);
//}
//void pushInstanceValue(lua_State *L, void *instanceKey, const char *name) {
//    lua_pushlightuserdata(L, instanceKey);
//    lua_gettable(L, LUA_REGISTRYINDEX);
//    lua_insert(L, -2);

//    lua_pushstring(L, name);
//    lua_insert(L, -2);

//    lua_settable(L, -3);
//    lua_remove(L, -1);
//}
//void getInstanceValue(lua_State *L, void *instanceKey, const char *name) {
//    lua_pushlightuserdata(L, instanceKey);
//    lua_gettable(L, LUA_REGISTRYINDEX);

//    lua_pushstring(L, name);

//    lua_gettable(L, -2);
//    lua_remove(L, -2);
//}
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

