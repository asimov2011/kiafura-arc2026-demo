"""A4 component support, bounding geometry, topology and extraction."""
from itertools import product
from .a4_core import *


def bbox_delta(g, color, diag, squares):
    support = set()
    for ps in components(g, diag=diag):
        missing = rect(ps)-ps
        if not missing:
            continue
        t,l,b,r = bbox(missing)
        if not squares or (b-t==r-l and len(missing)==(b-t+1)*(r-l+1)):
            support |= missing
    return paint(g,support,color)


def hollow_objects(g):
    support = set().union(*(ps-rect(ps,border=True) for ps in components(g)))
    return paint(g,support,modal(g))


def component_recolor(g, src, dst, kind, diag):
    selected = set()
    for ps in components(g,diag=diag,include_bg=True):
        if colorof(g,ps)!=src:
            continue
        t,l,b,r = bbox(ps)
        if ((kind=='non_singleton' and len(ps)>1) or
            (kind=='ring' and b>t and r>l and ps==rect(ps,border=True)) or
            (kind=='interior')):
            selected |= rect(ps,margin=-1) if kind=='interior' else ps
    return paint(g,selected,dst)


def region_classify(g, src, square_color, other_color):
    out = g
    for ps in components(g,include_bg=True):
        if colorof(g,ps)!=src:
            continue
        t,l,b,r=bbox(ps)
        out=paint(out,ps,square_color if b-t==r-l and len(ps)==(b-t+1)*(r-l+1) else other_color)
    return out


def enclosed_regions(g, src, inner, outer):
    h,w=shape(g)
    out=g
    for ps in components(g,include_bg=True):
        if colorof(g,ps)!=src:
            continue
        border=any(r in (0,h-1) or c in (0,w-1) for r,c in ps)
        out=paint(out,ps,outer if border else inner)
    return out


def nonsquare_interiors(g, src, dst):
    support=set()
    for ps in components(g):
        t,l,b,r=bbox(ps)
        if b-t!=r-l or len(ps)!=(b-t+1)*(r-l+1):
            support |= rect(ps,margin=-1)
    out=paint(g,cells(g,src),modal(g))
    return paint(out,support,dst)


def selected_crop(g, mono, diag, mode, value):
    comps=components(g,mono,diag)
    if mode=='palette_count':
        selected=[ps for ps in comps if len({g[r][c] for r,c in ps})==value]
    elif mode=='unique_shape':
        counts=Counter(norm(ps) for ps in comps)
        selected=[ps for ps in comps if counts[norm(ps)]==1]
    else:
        selected=[max(comps,key=len)]
    if len(selected)!=1:
        raise ValueError('nonunique selection')
    return crop(g,selected[0])


def foreground_quadrant(g):
    ps=set().union(*(set(p) for p in components(g,False,True)))
    x=crop(g,ps)
    return grid(row[:len(x[0])//2] for row in x[:len(x)//2])


def scaled_mask_crop(g, output_size):
    ps=max(components(g,diag=True),key=len)
    x=crop(g,ps)
    major=colorof(g,ps)
    minor=rare(x)
    x=tuple(tuple(0 if v==minor else minor if v==major else v for v in row) for row in x)
    h,w=shape(x)
    if h%output_size or w%output_size or h!=w:
        raise ValueError('nonintegral normalized block')
    step=h//output_size
    return grid(row[::step] for row in x[::step])


def unique_shape_recolor(g, common, unique, diag):
    comps=components(g,diag=diag)
    ct=Counter(norm(p) for p in comps)
    selected=[p for p in comps if ct[norm(p)]==min(ct.values())]
    if len(selected)!=1 or len(ct)<2:
        raise ValueError('no shape singleton')
    fg=set().union(*comps)
    return paint(paint(g,fg,common),selected[0],unique)


def centered_color_overlay(g):
    bg=modal(g)
    groups=[cells(g,c) for c in sorted(palette(g)-{bg})]
    extents=[(bbox(p)[2]-bbox(p)[0]+1,bbox(p)[3]-bbox(p)[1]+1) for p in groups]
    h,w=max(extents)
    out=canvas((h,w),bg)
    for p,(ph,pw) in zip(groups,extents):
        out=paint(out,shift(norm(p),(h-ph)//2,(w-pw)//2),colorof(g,p))
    return out


def symmetric_expanded_ring(g):
    comps=components(g,False)
    out=g
    for ps in comps:
        t,l,b,r=bbox(ps)
        if b-t!=r-l or b-t<2:
            raise ValueError('not a square frame')
        outer=g[t][l]
        inside=g[t+1][l+1]
        out=colored(out,{(i,j):(inside if g[i][j]==outer else outer) for i,j in ps})
        thickness=(r-l+1)-2
        expanded=rect(ps,thickness)
        corners={(t-thickness,l-thickness),(t-thickness,r+thickness),(b+thickness,l-thickness),(b+thickness,r+thickness)}
        forbidden=set()
        for cr,cc in corners:
            forbidden|={(i,j) for i in range(cr-thickness+1,cr+thickness) for j in range(cc-thickness+1,cc+thickness)}
        out=paint(out,expanded-forbidden,outer,under=True)
    return out


def frame_from_palette(g, ignore):
    bg=modal(g)
    remaining=palette(g)-{bg,ignore}
    if len(remaining)!=2:
        raise ValueError('frame palette')
    inner,outer=sorted(remaining,key=lambda c:len(cells(g,c)))
    fg=set().union(*(cells(g,c) for c in palette(g)-{bg}))
    return paint(paint(g,rect(fg),inner),rect(fg,border=True),outer)


def frontiers_recolor(g, color):
    ps={(r,c) for r,row in enumerate(g) if len(set(row))==1 for c in range(len(row))}
    ps|={(r,c) for c,col in enumerate(zip(*g)) if len(set(col))==1 for r in range(len(g))}
    if not ps:
        raise ValueError('no frontiers')
    return paint(g,ps,color)


def candidates(demos):
    def propose():
        yield hollow_objects,()
        yield foreground_quadrant,()
        yield centered_color_overlay,()
        yield symmetric_expanded_ring,()
        for ignore in colors(demos):
            yield frame_from_palette,(ignore,)
        a,b=demos[0]
        if len(b)==len(b[0]):
            yield scaled_mask_crop,(len(b),)
        for mono,diag in product((False,True),repeat=2):
            yield selected_crop,(mono,diag,'unique_shape',0)
            for n in sorted({len(palette(y)) for x,y in demos}):
                yield selected_crop,(mono,diag,'palette_count',n)
        for dst in added_colors(demos):
            yield frontiers_recolor,(dst,)
            for diag,squares in product((False,True),repeat=2):
                yield bbox_delta,(dst,diag,squares)
            for src in colors(demos):
                yield nonsquare_interiors,(src,dst)
                for diag in (False,True):
                    for kind in ('non_singleton','ring','interior'):
                        yield component_recolor,(src,dst,kind,diag)
        for src in colors(demos):
            for inside,outside in product(added_colors(demos),repeat=2):
                if inside==outside:
                    continue
                yield region_classify,(src,inside,outside)
                yield enclosed_regions,(src,inside,outside)
        for common,unique in product(added_colors(demos),repeat=2):
            if common!=unique:
                for diag in (False,True):
                    yield unique_shape_recolor,(common,unique,diag)
    return verified(demos,propose())
