// {{header1}}
// {{header2}}

#pragma once

{% set types = {
    'Long': 'long',
    'Float': 'float',
    'Double': 'double',
    'Byte': 'unsigned char'
}
%}

#include <iostream>
#include <string>

{% for Real in types %}
class TH{{Real}}Tensor;
class TH{{Real}}Storage;

int TH{{Real}}Storage_getRefCount(TH{{Real}}Storage *self);
int TH{{Real}}Tensor_getRefCount(TH{{Real}}Tensor *self);
{% endfor %}

struct lua_State;
lua_State *luaInit(void);
void luaClose(lua_State *L);
//void luaRequire(lua_State *L, const char *libName);

long pointerAsInt(void *ptr); // mostly for debugging
void collectGarbage(lua_State *L);

