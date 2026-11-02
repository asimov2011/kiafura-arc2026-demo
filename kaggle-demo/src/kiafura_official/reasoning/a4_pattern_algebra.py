"""A4 panel lattices, periodic masks, band reconstruction and self similarity."""
from itertools import product
from .a4_core import *


def runs(indices):
    result=[]
    for i in indices:
        if not result or i!=result[-1][-1]+1:
            result.append([i])
        else:
            result[-1].append(i)
    return result


def windows(g):
    rows=[r for r,row in enumerate(g) if len(set(row))>1]
    cols=[c for c,col in enumerate(zip(*g)) if len(set(col))>1]
    rr,cc=runs(rows),runs(cols)
    if len(rr)<2 or len(cc)<2:
        raise ValueError('two-dimensional lattice required')
    shapes={(len(r),len(c)) for r in rr for c in cc}
    if len(shapes)!=1:
        raise ValueError('inconsistent cells')
    return rr,cc


def lattice_max_fill(g):
    rr,cc=windows(g)
    col=rare(g)
    patches=[{(r,c) for r in rs for c in cs} for rs in rr for cs in cc]
    counts=[len(p&cells(g,col)) for p in patches]
    out=g
    for p,n in zip(patches,counts):
        out=paint(out,p,col if n==max(counts) else 0)
    return out


def lattice_count_reduce(g, src, dst, count):
    rr,cc=windows(g)
    return tuple(tuple(dst if sum(g[r][c]==src for r in rs for c in cs)==count else 0 for cs in cc) for rs in rr)


def lattice_marker_relocate(g, marker):
    rr,cc=windows(g)
    points=cells(g,marker)
    if len(points)!=1:
        raise ValueError('one marker required')
    mr,mc=next(iter(points))
    rs=next(rs for rs in rr if mr in rs)
    cs=next(cs for cs in cc if mc in cs)
    tr,tc=mr-rs[0],mc-cs[0]
    if tr>=len(rr) or tc>=len(cc):
        raise ValueError('marker does not name a cell')
    pattern=grid(tuple(g[r][c] for c in cs) for r in rs)
    support={(r,c) for rset in rr for cset in cc for r in rset for c in cset}
    out=paint(g,support,0)
    return colored(out,{(rr[tr][i],cc[tc][j]):v for i,row in enumerate(pattern) for j,v in enumerate(row)})


def sparse_half_frame(g):
    h,w=shape(g)
    if h%2:
        raise ValueError('even height required')
    out=g
    for half in range(2):
        part=g[half*(h//2):(half+1)*(h//2)]
        col=rare(part)
        ps=cells(part,col)
        if not ps or len({r for r,c in ps})!=1:
            raise ValueError('one horizontal marker row')
        row=next(iter(ps))[0]+half*(h//2)
        support={(row,c) for c in range(w)}|{(r,c) for r in range(half*(h//2),(half+1)*(h//2)) for c in (0,w-1)}
        edge=0 if half==0 else h-1
        support|={(edge,c) for c in range(w)}
        out=paint(out,support,col)
    return out


def side_band_recolor(g, src):
    comps=components(g)
    nv=sum(len({c for r,c in ps})==1 for ps in comps)
    nh=sum(len({r for r,c in ps})==1 for ps in comps)
    axis=1 if nv>nh else 0
    a,b=split_panels(g,axis,2)
    ca,cb=a[0][0],b[-1][-1]
    a=tuple(tuple(ca if v==src else v for v in row) for row in a)
    b=tuple(tuple(cb if v==src else v for v in row) for row in b)
    return tuple(ra+rb for ra,rb in zip(a,b)) if axis else a+b


def band_hole_extend(g, hole):
    horizontal=any(len(set(row))==1 for row in g)
    x=g if horizontal else transpose(g)
    out=x
    for col in sorted(palette(x)-{hole}):
        ps=cells(x,col)
        t,l,b,r=bbox(ps)
        holes={(i,j) for i in range(t,b+1) for j in range(l,r+1) if x[i][j]==hole}
        out=paint(out,{(i,j) for _,j in holes for i in range(t,b+1) if (i,j) in ps},hole)
    return out if horizontal else transpose(out)


def band_majority(g):
    rows=[modal((row,)) for row in g]
    cols=[modal((col,)) for col in zip(*g)]
    if len(set(cols))>len(set(rows)):
        return tuple(tuple(cols) for _ in g)
    return tuple((c,)*len(g[0]) for c in rows)


def quadrant_reflect_compress(g):
    h,w=shape(g)
    midr,midc=h//2,w//2
    sep=g[midr][midc]
    if any(v!=sep for v in g[midr]) or any(row[midc]!=sep for row in g):
        raise ValueError('cross separator missing')
    bg=modal(g)
    out=canvas((h-1,w-1),bg)
    assignments={}
    for r,row in enumerate(g):
        for c,v in enumerate(row):
            if r==midr or c==midc or v in (bg,sep):
                continue
            rr=r if r<midr else r-1
            cc=c if c<midc else c-1
            for i,j in ((rr,cc),(h-2-rr,cc),(rr,w-2-cc),(h-2-rr,w-2-cc)):
                assignments[i,j]=sep
    return colored(out,assignments)


def self_mask_tile(g, use_modal):
    h,w=shape(g)
    if h!=w or h*h>60:
        raise ValueError('square mask required')
    bg=modal(g)
    return tuple(tuple(g[r%h][c%w] if ((g[r//h][c//w]==bg) if use_modal else (g[r//h][c//w]!=bg)) else 0
                       for c in range(w*w)) for r in range(h*h))


def reduced_self_mask(g, factor):
    ps=set().union(*components(g,diag=True))
    x=crop(g,ps)
    if len(x)%factor or len(x[0])%factor:
        raise ValueError('block shape')
    small=grid(row[::factor] for row in x[::factor])
    h,w=shape(small)
    if max(h*h,w*w)>60:
        raise ValueError('self mask extent')
    bg=modal(g)
    return tuple(tuple(small[r%h][c%w] if small[r//h][c//w]!=bg else bg for c in range(w*w)) for r in range(h*h))


def panel_kronecker(g):
    parts=split_panels(g,0 if len(g)>len(g[0]) else 1,2)
    mask=min(parts,key=lambda p:len(palette(p)))
    pattern=max(parts,key=lambda p:len(palette(p)))
    h,w=shape(pattern)
    if max(h*h,w*w)>60:
        raise ValueError('product extent')
    return tuple(tuple(pattern[r//h][c//w] if mask[r%h][c%w]!=0 else 0 for c in range(w*w)) for r in range(h*h))


def corner_palette_mask(g, marker):
    ps=cells(g,marker)
    x=crop(g,ps)
    h,w=shape(x)
    corners=(g[0][0],g[0][-1],g[-1][0],g[-1][-1])
    if h%2 or w%2:
        raise ValueError('even mask required')
    return tuple(tuple(corners[2*(r>=h//2)+(c>=w//2)] if x[r][c]!=0 else 0 for c in range(w)) for r in range(h))


def candidates(demos):
    def propose():
        for fn in (lattice_max_fill,sparse_half_frame,band_majority,quadrant_reflect_compress,panel_kronecker):
            yield fn,()
        yield self_mask_tile,(True,)
        for factor in (2,3):
            yield reduced_self_mask,(factor,)
        for src in colors(demos):
            yield lattice_marker_relocate,(src,)
            yield side_band_recolor,(src,)
            yield band_hole_extend,(src,)
            yield corner_palette_mask,(src,)
            # Counts are witnessed in the positively marked first-demo cells.
            a,b=demos[0]
            try:
                rr,cc=windows(a)
                if shape(b)!=(len(rr),len(cc)):
                    continue
                for dst in added_colors(demos):
                    counts={sum(a[r][c]==src for r in rr[i] for c in cc[j]) for i,row in enumerate(b) for j,v in enumerate(row) if v==dst}
                    if len(counts)==1:
                        yield lattice_count_reduce,(src,dst,counts.pop())
            except ERRORS:
                pass
    return verified(demos,propose())
