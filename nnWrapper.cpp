#include "nnWrapper.h"

#include "THTensor.h"
//#include "lual.h"
//#in

extern "C" {
    #include "lua.h"
    #include "lauxlib.h"
    #include "lualib.h"
}

#ifndef _WIN32
    #include <dlfcn.h>
#endif

#include <iostream>
#include <stdexcept>
using namespace std;

lua_State *xluaInit() {
    #ifndef _WIN32
    void *hdl = dlopen("liblua5.1.so", RTLD_NOW | RTLD_GLOBAL);
    if(hdl == 0) {
        cout << dlerror() << endl;
        throw runtime_error(string("Couldnt load liblua5.1.so ") + dlerror());
//        return 0;
    } else {
        cout << "loaded lua library" << endl;
    }
    #endif

    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    lua_getglobal(L, "require");
    lua_pushstring(L, "paths");
//    lua_pushstring(L, "nn");
    lua_call(L, 1, 1);
//    lua_pushliteral(L, "nn");
//    luaopen_nn(L);
    lua_setglobal(L, "nn");
    return L;
}
void xluaClose(lua_State *L) {
    lua_close(L);
}

