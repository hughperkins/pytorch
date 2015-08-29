#include "nnWrapper.h"

#include "THTensor.h"
//#include "lual.h"
//#in
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include <iostream>
using namespace std;

lua_State *luaInit() {
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    lua_getglobal(L, "require");
    lua_pushstring(L, "nn");
    lua_call(L, 1, 1);
//    lua_pushliteral(L, "nn");
//    luaopen_nn(L);
    lua_setglobal(L, "nn");
    return L;
}
void luaClose(lua_State *L) {
    lua_close(L);
}

