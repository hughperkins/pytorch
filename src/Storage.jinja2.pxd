# {{header1}}
# {{header2}}

{% set types = {
    'Long': {'real': 'long'},
    'Float': {'real': 'float'}, 
    'Double': {'real': 'double'},
    'Byte': {'real': 'unsigned char'}
}
%}

{% for Real in types %}
cdef extern from "nnWrapper.h":
    int TH{{Real}}Storage_getRefCount(TH{{Real}}Storage *self)
{% endfor %}

{% for Real in types %}
{% set real = types[Real]['real'] %}
cdef extern from "THStorage.h":
    cdef struct TH{{Real}}Storage
    TH{{Real}}Storage* TH{{Real}}Storage_newWithData({{real}} *data, long size)
    TH{{Real}}Storage* TH{{Real}}Storage_new()
    TH{{Real}}Storage* TH{{Real}}Storage_newWithSize(long size)
    {{real}} *TH{{Real}}Storage_data(TH{{Real}}Storage *self)
    long TH{{Real}}Storage_size(TH{{Real}}Storage *self)
    void TH{{Real}}Storage_free(TH{{Real}}Storage *self)
    void TH{{Real}}Storage_retain(TH{{Real}}Storage *self)
    void TH{{Real}}Storage_set(TH{{Real}}Storage*, long, {{real}})
    {{real}} TH{{Real}}Storage_get(const TH{{Real}}Storage*, long)

{% endfor %}

{% for Real in types %}
{% set real = types[Real]['real'] %}
cdef class _{{Real}}Storage(object):
    cdef TH{{Real}}Storage *native
    cpdef long size(self)

cdef _{{Real}}Storage_fromNative(TH{{Real}}Storage *storageC, retain=*)
{% endfor %}

