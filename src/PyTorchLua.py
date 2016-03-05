import types

#def makeClass(base, path, name):
##    class Foo(object):
##        def __init__(self):
##            pass
##    Foo.__class__.__name__ = name
#    meta = type(base)
#    meta = types._calculate_meta(meta, (base,))
#    print('path', path)
#    ns = meta.__prepare__(name, (base,), {})
#    return meta(path + '.' + name, (base,), ns)
##    RenamedFoo = types.new_class(path + '.' + name, (Foo,))
##    return RenamedFoo

def renameClass(base, path, name):
    return type(base)(name, (base,), {})
#    meta = type(base)
#    meta = types._calculate_meta(meta, (base,))
#    print('path', path)
#    ns = meta.__prepare__(name, (base,), {})
#    return meta(path + '.' + name, (base,), ns)

