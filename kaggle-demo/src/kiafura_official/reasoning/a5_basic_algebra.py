"""A5 geometry, pointwise color, scale, periodic layout: cohort-derived graphs."""
from .a5_core import *


def rigid(g,k):
    return transform(g,k)


def recolor(g, mapping):
    d=dict(mapping)
    return tuple(tuple(d.get(c,c) for c in row) for row in g)


def tile_transforms(g,nr,nc,ks):
    tiles=[transform(g,k) for k in ks]
    return sum((hcat(tiles[r*nc:(r+1)*nc]) for r in range(nr)),())


def palette_scale(g):
    return scale(g,len(palette(g)))


def clean_downscale(g,n,noise,background):
    g=fill(g,background,cells(g,noise))
    if len(g)%n or len(g[0])%n:
        raise ValueError('scale')
    return tuple(tuple(row[::n]) for row in g[::n])


def column_sort(g,descending):
    return grid(zip(*(sorted(col,reverse=descending) for col in zip(*g))))


def self_mask(g,mask,background):
    if mask=='least': mask=min(palette(g),key=lambda c:len(cells(g,c)))
    h,w=shape(g)
    return tuple(tuple(g[r%h][c%w] if g[r//h][c//w]==mask else background for c in range(w*w)) for r in range(h*h))


def parity_upscale(g,mask,a,b,background,n):
    q=scale(g,n)
    return tuple(tuple((a if (r+c)%2==0 else b) if v==mask else background for c,v in enumerate(row)) for r,row in enumerate(q))


def half_reflection(g,axis,part):
    if axis: g=transform(g,5)
    h=len(g)//2
    q=g[:h] if part==0 else g[-h:]
    out=q+q[::-1] if part==0 else q[::-1]+q
    return transform(out,5) if axis else out


def oscillating_rows(g):
    period=g+g[-2::-1]
    return period+period[1:]


def row_seed_expand(g):
    h,w=shape(g)
    if h!=2*w+2:
        raise ValueError('layout')
    return g[:2]+tuple((c,)*w for _ in range(2) for c in g[0])


def deduplicate_axes(g):
    rows=tuple(dict.fromkeys(g))
    return grid(zip(*dict.fromkeys(zip(*rows))))


def candidates(demos):
    if not demos:return ()
    result=[]
    for k in range(8): offer(result,demos,rigid,k)
    if same_shape(demos):
        mapping={}
        valid=True
        for i,o in demos:
            for ir,orr in zip(i,o):
                for a,b in zip(ir,orr):
                    if a in mapping and mapping[a]!=b: valid=False
                    mapping[a]=b
        if valid: offer(result,demos,recolor,tuple(sorted(mapping.items())))
        for reverse in (False,True): offer(result,demos,column_sort,reverse)
        for axis,part in product(range(2),repeat=2): offer(result,demos,half_reflection,axis,part)
    # Bind each tile's orientation independently against ALL demos; no tile grids retained.
    hi,wi=shape(demos[0][0]); ho,wo=shape(demos[0][1])
    if ho%hi==0 and wo%wi==0 and 1<ho//hi*(wo//wi)<=4:
        nr,nc=ho//hi,wo//wi
        if all(shape(o)==(len(i)*nr,len(i[0])*nc) for i,o in demos):
            choices=[]
            for r,c in product(range(nr),range(nc)):
                choices.append([k for k in range(8) if all(transform(i,k)==tuple(tuple(row[c*len(i[0]):(c+1)*len(i[0])]) for row in o[r*len(i):(r+1)*len(i)]) for i,o in demos)])
            for ks in product(*choices): offer(result,demos,tile_transforms,nr,nc,ks)
    offer(result,demos,palette_scale)
    offer(result,demos,oscillating_rows)
    offer(result,demos,row_seed_expand)
    offer(result,demos,deduplicate_axes)
    if ho<hi and wo<wi:
        for n in range(2,5):
            for noise,background in product(colors(demos),colors(demos,1)):
                offer(result,demos,clean_downscale,n,noise,background)
    if ho==hi*hi and wo==wi*wi:
        for mask,background in product(colors(demos)+['least'],colors(demos,1)):
            offer(result,demos,self_mask,mask,background)
    if ho==2*hi and wo==2*wi:
        for mask in colors(demos):
            for a,b,background in permutations(colors(demos,1),3):
                offer(result,demos,parity_upscale,mask,a,b,background,2)
    return tuple(result)
