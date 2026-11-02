"""A4 coordinate, pointwise, panel and count/render families."""
from itertools import product
from .a4_core import *


def isometry(g, turns, mirror):
    return rotate(tuple(row[::-1] for row in g) if mirror else g, turns)


def pointwise(g, mapping):
    m = dict(mapping)
    return tuple(tuple(m.get(v,v) for v in row) for row in g)


def cell_scale(g, rh, cw):
    return scale(g,rh,cw)


def sample_scale(g, factor, offset, expansion):
    return scale(grid(row[offset::factor] for row in g[offset::factor]), expansion)


def panel_select(g, axis, n, index):
    return split_panels(g,axis,n)[index]


def panel_boolean(g, axis, ca, cb, table, reflect):
    a,b = split_panels(g,axis,2)
    if reflect:
        b = tuple(row[::-1] for row in b)
    return tuple(tuple(table[2*(x==ca)+(y==cb)] for x,y in zip(ra,rb)) for ra,rb in zip(a,b))


def panel_overlay(g, axis, n, bg, reflect):
    parts = split_panels(g,axis,n)
    out = parts[0]
    for b in parts[1:]:
        if reflect:
            b = tuple(row[::-1] for row in b)
        out = tuple(tuple(x if x != bg else y for x,y in zip(a,b)) for a,b in zip(out,b))
    return out


def modal_canvas(g, output_shape):
    return canvas(output_shape, modal(g))


def count_diagonal(g, color):
    n = len(components(g))
    return paint(canvas((n,n)), {(i,i) for i in range(n)},color)


def count_bar_overflow(g, src, dst, outshape):
    n = len(cells(g,src))
    h,w = outshape
    ps = {(0,i) for i in range(min(w,n))}
    ps |= {(r,w//2) for r in range(1,n-w+1)}
    return paint(canvas(outshape),ps,dst)


def rank_color_strip(g, diag, reverse):
    comps = sorted(components(g,diag=diag),key=lambda p:(len(p),min(p)),reverse=reverse)
    return tuple((colorof(g,p),) for p in comps)


def reflected_tile(g, nr, nc, vertical_phase, horizontal_phase):
    h,w = shape(g)
    return tuple(tuple(g[(h-1-r%h) if (r//h+vertical_phase)%2 else r%h]
                        [(w-1-c%w) if (c//w+horizontal_phase)%2 else c%w]
                        for c in range(w*nc)) for r in range(h*nr))


def diagonal_period(g, period, slope):
    bg = modal(g)
    binding = {}
    for r,row in enumerate(g):
        for c,v in enumerate(row):
            if v == bg:
                continue
            k = (r+slope*c)%period
            if k in binding and binding[k] != v:
                raise ValueError('inconsistent diagonal colors')
            binding[k] = v
    if len(binding) != period:
        raise ValueError('unobserved phase')
    return tuple(tuple(binding[(r+slope*c)%period] for c in range(len(g[0]))) for r in range(len(g)))


def candidates(demos):
    def propose():
        for turns,mirror in product(range(4),range(2)):
            yield isometry,(turns,mirror)
        if all(shape(a)==shape(b) for a,b in demos):
            m = {}
            ok = True
            for a,b in demos:
                for ra,rb in zip(a,b):
                    for x,y in zip(ra,rb):
                        if x in m and m[x] != y:
                            ok = False
                        m[x] = y
            if ok:
                yield pointwise,(tuple(sorted(m.items())),)
            for period in range(2,1+len(colors(demos))):
                for slope in (-1,1):
                    yield diagonal_period,(period,slope)
        a,b = demos[0]
        h,w = shape(a)
        oh,ow = shape(b)
        if oh%h == 0 and ow%w == 0:
            rh,cw = oh//h,ow//w
            yield cell_scale,(rh,cw)
            for pv,ph in product(range(2),repeat=2):
                yield reflected_tile,(rh,cw,pv,ph)
        for down in (2,3):
            if h%down==0 and w%down==0 and oh%(h//down)==0 and ow%(w//down)==0:
                expansion = oh//(h//down)
                if expansion == ow//(w//down):
                    for offset in range(down):
                        yield sample_scale,(down,offset,expansion)
        if all(shape(y)==shape(b) for x,y in demos):
            yield modal_canvas,(shape(b),)
            for src,dst in product(colors(demos),added_colors(demos)):
                yield count_bar_overflow,(src,dst,shape(b))
        for color in colors(demos,True):
            yield count_diagonal,(color,)
        for diag,reverse in product(range(2),repeat=2):
            yield rank_color_strip,(diag,reverse)
        for axis,n in product(range(2),(2,3)):
            try:
                parts = split_panels(a,axis,n)
            except ERRORS:
                continue
            if shape(parts[0]) != shape(b):
                continue
            yield panel_select,(axis,n,0)
            for bg,reflect in product(colors(demos),(False,True) if n==2 else (False,)):
                yield panel_overlay,(axis,n,bg,reflect)
            if n != 2:
                continue
            for ca,cb,reflect in product(colors(demos),colors(demos),(False,True)):
                table = {}
                ok = True
                for x,y in demos:
                    try:
                        p,q = split_panels(x,axis,2)
                        if reflect:
                            q = tuple(row[::-1] for row in q)
                        if shape(p)!=shape(y):
                            ok=False
                            break
                        for rp,rq,ry in zip(p,q,y):
                            for u,v,z in zip(rp,rq,ry):
                                k=2*(u==ca)+(v==cb)
                                if k in table and table[k]!=z:
                                    ok=False
                                table[k]=z
                    except ERRORS:
                        ok=False
                if ok and len(table)==4:
                    yield panel_boolean,(axis,ca,cb,tuple(table[i] for i in range(4)),reflect)
    return verified(demos,propose())
