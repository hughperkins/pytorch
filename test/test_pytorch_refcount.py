from __future__ import print_function, division
import PyTorch
import array
import numpy
import inspect
import gc


def myeval(expr):
    l = locals()
    context = inspect.getouterframes( inspect.currentframe() )[1].frame.f_locals
    for k, v in context.items():
        l[k] = v
    print(expr, eval(expr))


def test_refcount():
    D = PyTorch.FloatTensor(1000, 1000).fill(1)
    myeval('D.isContiguous()')
    myeval('D.refCount')
    assert D.refCount == 1

    print('\nget storage into Ds')
    Ds = D.storage()
    myeval('D.refCount')
    myeval('Ds.refCount')
    assert D.refCount == 1
    assert Ds.refCount == 2

    print('\nget E')
    E = D.narrow(1, 100, 800)
    myeval('Ds.refCount')
    myeval('E.isContiguous()')
    myeval('D.refCount')
    myeval('E.refCount')
    assert Ds.refCount == 3
    assert E.refCount == 1
    assert D.refCount == 1

    print('\nget Es')
    Es = E.storage()
    myeval('Ds.refCount')
    myeval('Es.refCount')
    myeval('E.isContiguous()')
    myeval('D.refCount')
    myeval('E.refCount')
    assert Es.refCount == 4
    assert Ds.refCount == 4
    assert E.refCount == 1
    assert D.refCount == 1

    print('\nget Ec')
    Ec = E.contiguous()
    myeval('Ds.refCount')
    myeval('Es.refCount')
    myeval('D.refCount')
    myeval('E.refCount')
    myeval('Ec.refCount')

    assert Es.refCount == 4
    assert Ds.refCount == 4
    assert E.refCount == 1
    assert D.refCount == 1

    assert Ec.refCount == 1

    print('\nget Ecs')
    Ecs = Ec.storage()
    myeval('Ds.refCount')
    myeval('Es.refCount')
    myeval('D.refCount')
    myeval('E.refCount')
    myeval('Ec.refCount')
    myeval('Ecs.refCount')

    assert Es.refCount == 4
    assert Ds.refCount == 4
    assert E.refCount == 1
    assert D.refCount == 1

    assert Ec.refCount == 1
    assert Ecs.refCount == 2

    Dc = D.contiguous()
    print('\nafter creating Dc')
    myeval('D.refCount')
    myeval('Dc.refCount')
    myeval('Ds.refCount')
    assert D.refCount == 2
    assert Dc.refCount == 2
    assert Ds.refCount == 4

    Dcs = Dc.storage()
    print('\n after get Dcs')
    assert D.refCount == 2
    assert Dc.refCount == 2
    myeval('Ds.refCount')
    myeval('Dcs.refCount')
    assert Ds.refCount == 5
    assert Dcs.refCount == 5

    D = None
    E = None
    Ec = None
    Dc = None
    gc.collect()
    print('\n after setting tensors to None')
    myeval('Ds.refCount')
    myeval('Es.refCount')
    myeval('Ecs.refCount')

    assert Dcs.refCount == 3
    assert Ds.refCount == 3
    assert Es.refCount == 3
    assert Ecs.refCount == 1
