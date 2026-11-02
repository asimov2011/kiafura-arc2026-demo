"""A4 connection, axial/diagonal continuation and marker fields."""
from itertools import product
from .a4_core import *


def color_closure(g):
    out=g
    for col in sorted(palette(g)-{modal(g)}):
        ps=cells(g,col)
        support=set().union(*(line(a,b) for a in ps for b in ps))
        out=paint(out,support,col)
    return out


def orthogonal_connection(g, src, dst):
    ps=cells(g,src)
    if len(ps)<2:
        raise ValueError('no pair')
    support=set().union(*(line(a,b) for a in ps for b in ps if a[0]==b[0] or a[1]==b[1]))
    return paint(g,support,dst,under=True)


def component_cross(g, dst):
    h,w=shape(g)
    support=set()
    for ps in components(g):
        r,c=center(ps)
        support |= {(r,i) for i in range(w)}|{(i,c) for i in range(h)}
    return paint(g,support,dst,under=True)


def color_axes(g, vertical_color):
    h,w=shape(g)
    out=g
    vertical=cells(g,vertical_color)
    out=paint(out,{(r,c) for _,c in vertical for r in range(h)},vertical_color)
    for ps in components(g):
        col=colorof(g,ps)
        if col!=vertical_color:
            r,_=center(ps)
            out=paint(out,{(r,c) for c in range(w)},col)
    return out


def outward_corner_rays(g, src, dr, dc):
    ps=cells(g,src)
    t,l,b,r=bbox(ps)
    p=(b if dr>0 else t,r if dc>0 else l)
    return paint(g,ray(g,p,(dr,dc)),src)


def paired_corner_rays(g, lower_color, upper_color):
    out=outward_corner_rays(g,lower_color,1,1)
    return outward_corner_rays(out,upper_color,-1,-1)


def marker_centroid_ray(g):
    col=rare(g)
    mark=cells(g,col)
    foreground=set().union(*components(g))
    start=center(mark)
    target=center(foreground)
    direction=target[0]-start[0],target[1]-start[1]
    return paint(g,ray(g,start,direction),col,under=True)


def bbox_outward_diagonals(g):
    comps=components(g,diag=True)
    ps=max(comps,key=lambda p:(bbox(p)[2]-bbox(p)[0]+1)*(bbox(p)[3]-bbox(p)[1]+1))
    base=colorof(g,ps)
    others=palette(g)-{modal(g),base}
    if len(others)!=1:
        raise ValueError('ray color ambiguous')
    t,l,b,r=bbox(ps)
    support=set().union(*(ray(g,p,d) for p,d in [((t,l),(-1,-1)),((t,r),(-1,1)),((b,l),(1,-1)),((b,r),(1,1))]))
    return paint(g,support,next(iter(others)),under=True)


def corner_labels(g, labels):
    out=g
    for ps in components(g):
        t,l,b,r=bbox(ps)
        out=colored(out,{(t-1,l-1):labels[0],(t-1,r+1):labels[1],(b+1,l-1):labels[2],(b+1,r+1):labels[3]})
    return out


def diagonal_canvas(g, ratio):
    out=canvas((len(g)*ratio,len(g[0])*ratio),modal(g))
    for ps in components(g):
        out=paint(out,ray(out,center(ps),(1,1)),colorof(g,ps))
    return out


def rare_column_tile(g, dst, factor):
    ps=cells(g,rare(g))
    out=paint(g,{(r,c) for _,c in ps for r in range(len(g))},dst,under=True)
    return tuple(row*factor for _ in range(factor) for row in out)


def border_pair_connections(g):
    col=rare(g)
    ps=cells(g,col)
    fg=set().union(*components(g))
    boundary=rect(fg,border=True)
    support=set()
    for a in ps:
        for b in ps:
            if a[0]!=b[0] and a[1]!=b[1]:
                continue
            segment=line(a,b)
            if segment-boundary:
                support |= segment
    return paint(canvas(shape(g),modal(g)),support,col)


def directional_wedge(g, src, above, below):
    t,l,b,r=bbox(cells(g,src))
    pivot=(t,r)
    out=g
    for row in range(len(g)):
        boundary=pivot[1]+pivot[0]-row
        col=above if row<pivot[0] else below
        out=paint(out,{(row,c) for c in range(min(len(g[0]),boundary+1))},col,under=True)
    return out


def broken_frame_ray(g, src, dst):
    ps=cells(g,src)
    frame=rect(ps,border=True)
    missing=frame-ps
    if not missing:
        raise ValueError('closed frame')
    cr,cc=center(frame)
    mr,mc=center(missing)
    d=((mr>cr)-(mr<cr),(mc>cc)-(mc<cc))
    return paint(paint(g,rect(ps)-ps,dst),ray(g,min(missing),d),dst)


def cross_shift_by_count(g, marker):
    bg=modal(g)
    n=len(cells(g,marker))
    x=paint(g,cells(g,marker),bg)
    col=rare(x)
    candidates=[(r,c) for r,c in cells(x,col) if all((r+dr,c+dc) in cells(x,col) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))]
    if len(candidates)!=1:
        raise ValueError('cross center ambiguous')
    r,c=candidates[0]
    r,c=r+n,c-n
    return paint(canvas(shape(g),bg),{(r,i) for i in range(len(g[0]))}|{(i,c) for i in range(len(g))},col)


def candidates(demos):
    def propose():
        for fn in (color_closure,marker_centroid_ray,bbox_outward_diagonals,border_pair_connections):
            yield fn,()
        srcs=colors(demos)
        dsts=added_colors(demos)
        for src in srcs:
            yield color_axes,(src,)
            yield cross_shift_by_count,(src,)
            for dst in dsts:
                yield orthogonal_connection,(src,dst)
                yield broken_frame_ray,(src,dst)
            for other in srcs:
                if other!=src:
                    yield paired_corner_rays,(src,other)
        for dst in dsts:
            yield component_cross,(dst,)
            a,b=demos[0]
            if len(b)%len(a)==0 and len(b[0])//len(a[0])==len(b)//len(a):
                yield rare_column_tile,(dst,len(b)//len(a))
        for src,above,below in product(srcs,dsts,dsts):
            if above!=below:
                yield directional_wedge,(src,above,below)
        a,b=demos[0]
        if len(b)%len(a)==0:
            yield diagonal_canvas,(len(b)//len(a),)
        # Four corner labels are read at geometrically determined demo positions.
        labels=None
        try:
            ps=components(a)[0]
            t,l,bt,r=bbox(ps)
            points=((t-1,l-1),(t-1,r+1),(bt+1,l-1),(bt+1,r+1))
            if any(i<0 or j<0 for i,j in points):
                raise ValueError('corner outside')
            labels=tuple(b[i][j] for i,j in points)
        except ERRORS:
            pass
        if labels:
            yield corner_labels,(labels,)
    return verified(demos,propose())
