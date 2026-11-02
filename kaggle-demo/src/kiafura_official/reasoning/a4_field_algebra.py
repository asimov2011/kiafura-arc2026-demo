"""A4 marker-controlled fields, bounded corridors and ray endpoint rendering."""
from itertools import product
from .a4_core import *
from .a4_move_algebra import contact_delta


def palette_scale_diagonals(g,dst):
    factor=len(palette(g))-1
    x=scale(g,factor)
    ps=min(components(x,False),key=len)
    t,l,b,r=bbox(ps)
    support=set().union(*(ray(x,p,d) for p,d in [((t,l),(-1,-1)),((t,l),(1,1)),((b,l),(1,-1)),((b,l),(-1,1))]))
    return paint(x,support,dst,under=True)


def paired_bar_echo(g,marker,bar,replacement):
    bars=[p for p in components(g) if colorof(g,p)==bar]
    if len(bars)!=2:
        raise ValueError('two bars required')
    bars.sort(key=lambda p:bbox(p)[0])
    delta=bbox(bars[1])[0]-bbox(bars[0])[0]
    direction=-1 if bbox(bars[0])[1]==0 else 1
    marks=cells(g,marker)
    out=paint(g,marks,replacement)
    for r,c in marks:
        out=paint(out,ray(g,(r,c),(0,direction)),marker,under=True)
        out=paint(out,{(r+delta,j) for j in range(len(g[0]))},marker,under=True)
    return out


def marker_thick_ray(g,marker,body):
    centerlines=set(); bands=set()
    for ps in components(g,False):
        marks=ps&cells(g,marker)
        mt,ml,mb,mr=bbox(marks)
        t,l,b,r=bbox(ps)
        dr=int(mb==b)-int(mt==t)
        dc=int(mr==r)-int(ml==l)
        beam=ray(g,center(marks),(dr,dc))
        thickness=min(b-t+1,r-l+1)-1
        centerlines|=beam
        for offset in range(-thickness,thickness+1):
            bands|=shift(beam,0,offset) if dr else shift(beam,offset,0)
    return paint(paint(g,centerlines,marker),bands,body,under=True)


def seeded_maze_parity(g):
    seedcolor=rare(g)
    seeds=cells(g,seedcolor)
    p=min(seeds)
    adjacent=[g[p[0]+dr][p[1]+dc] for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))
              if 0<=p[0]+dr<len(g) and 0<=p[1]+dc<len(g[0])]
    if not adjacent:
        raise ValueError('seed has no neighbors')
    ct=Counter(adjacent);other=min(ct,key=lambda c:(-ct[c],c))
    boundary=cells(g,other)
    selected=set()
    for ps in components(g,include_bg=True):
        if colorof(g,ps)!=0:
            continue
        if any((r+dr,c+dc) in boundary for r,c in ps for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))):
            selected|=ps
    return colored(g,{(r,c):seedcolor if min(abs(r-i)+abs(c-j) for i,j in seeds)%2==0 else other for r,c in selected})


def diagonal_color_fields(g):
    comps=components(g,include_bg=True)
    singles=[p for p in comps if len(p)==1]
    if len(singles)<2:
        raise ValueError('multiple field markers required')
    mapping={}
    support=set()
    for p in singles:
        r,c=next(iter(p));col=colorof(g,p)
        neighbors=[g[r+dr][c+dc] for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and 0<=r+dr<len(g) and 0<=c+dc<len(g[0])]
        ct=Counter(neighbors);base=min(ct,key=lambda c:(-ct[c],c))
        if base in mapping and mapping[base]!=col:
            raise ValueError('inconsistent field markers')
        mapping[base]=col
        for d in ((1,1),(1,-1),(-1,1),(-1,-1)):
            support|=ray(g,(r,c),d)
    return colored(g,{(r,c):mapping[g[r][c]] for r,c in support if g[r][c] in mapping})


def bounded_corridor(g,dst):
    comps=components(g)
    if len(comps)!=2:
        raise ValueError('two corridor endpoints required')
    small,big=sorted(comps,key=len)
    t,l,b,r=bbox(small);bt,bl,bb,br=bbox(big)
    vertical=bool({c for i,c in small}&{c for i,c in big})
    direction=(-1 if t>bt else 1,0) if vertical else (0,-1 if l>bl else 1)
    # The inner frame remains a line for a two-cell-thick endpoint.
    inner={(i,j) for i in range(min(t+1,b-1),max(t+1,b-1)+1) for j in range(min(l+1,r-1),max(l+1,r-1)+1)}
    beams=set().union(*(ray(g,p,direction) for p in inner))
    provisional=paint(g,beams,dst,under=True)
    reject=set()
    for ps in components(provisional):
        if colorof(provisional,ps)==dst and any(i in (0,len(g)-1) or j in (0,len(g[0])-1) for i,j in ps):
            reject|=ps
    return paint(provisional,reject,modal(g))


def contact_endpoint_frame(g,mover,bar,max_length):
    movers=[p for p in components(g) if colorof(g,p)==mover]
    bars=[p for p in components(g) if colorof(g,p)==bar]
    support=set();endpoints=set()
    for a in movers:
        for b in bars:
            try:
                dr,dc=contact_delta(a,b)
            except ValueError:
                continue
            dr+=(dr>0)-(dr<0);dc+=(dc>0)-(dc<0)
            start=center(a);end=(start[0]+dr,start[1]+dc)
            segment=line(start,end)
            if len(segment)<max_length:
                support|=segment;endpoints.add(end)
    if not endpoints:
        raise ValueError('no reachable short contact')
    frame={(r+dr,c+dc) for r,c in endpoints for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc}
    return paint(paint(g,support,mover),frame,bar)


def candidates(demos):
    def propose():
        yield seeded_maze_parity,()
        yield diagonal_color_fields,()
        cols=colors(demos)
        new=added_colors(demos)
        for dst in new:
            yield palette_scale_diagonals,(dst,)
            yield bounded_corridor,(dst,)
        for marker,bar in product(cols,repeat=2):
            if marker==bar:
                continue
            yield marker_thick_ray,(marker,bar)
            yield contact_endpoint_frame,(marker,bar,8)
            for replacement in new:
                yield paired_bar_echo,(marker,bar,replacement)
    return verified(demos,propose())
