# {{header1}}
# {{header2}}

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
cdef extern from "nnWrapper.h":
    int TH{{Real}}Storage_getRefCount(TH{{Real}}Storage *self)
{% endfor %}

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
{% if Real == 'Cl' %}
cdef extern from "THCl/THClStorage.h":
{% else %}
cdef extern from "THStorage.h":
{% endif %}
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

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
cdef class _{{Real}}Storage(object):
    cdef TH{{Real}}Storage *native
    cpdef long size(self)

cdef _{{Real}}Storage_fromNative(TH{{Real}}Storage *storageC, retain=*)
{% endfor %}

