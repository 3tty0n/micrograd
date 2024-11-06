import py
py.path.local(__file__)

from micrograd.engine import Value

def entry_point(args):
    a = Value(-4.0)
    b = Value(2.0)
    c = a.add(b)
    d = a .mul(b).add(b.pow(3.0))
    c = c.add(c.add(Value(1)))
    c = c.add(Value(1.0)).add(a.mul(Value(-1)))
    d = d.mul(Value(2.0)).add((b.add(a)).relu()).add(d)
    d = d.mul(Value(3.0)).add((b.add(a.mul(Value(-1.0)))).relu()).add(d)
    e = c.add(d.mul(Value(-1.0)))
    f = e.pow(2.0)
    f.backward()
    # g = f / 2
    # g = Value(10.0) / f
    # print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
    # g.backward()
    # print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
    # print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
    return 0


def target(driver, args):
    return entry_point


if __name__ == '__main__':
    import sys
    sys.exit(entry_point(sys.argv))
