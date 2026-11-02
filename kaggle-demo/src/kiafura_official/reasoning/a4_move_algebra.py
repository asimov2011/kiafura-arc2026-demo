"""A4 translation, contact, docking, sorting and reflection completion."""
from itertools import product, permutations
from .a4_core import *


def contact_delta(a,b):
    options=[]
    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
        for n in range(61):
            moved=shift(a,dr*n,dc*n)
            if moved&b:
                break
            if any((r+ar,c+ac) in b for r,c in moved for ar,ac in ((1,0),(-1,0),(0,1),(0,-1))):
                options.append((n,dr*n,dc*n))
                break
    if not options:
        raise ValueError('no axial contact')
    _,dr,dc=min(options)
    return dr,dc


def height_rise(g):
    out=canvas(shape(g),modal(g))
    for ps in components(g):
        t,l,b,r=bbox(ps)
        out=paint(out,shift(ps,-(b-t+1),0),colorof(g,ps))
    return out


def sorted_bars(g):
    out=canvas(shape(g),modal(g))
    ps=sorted(components(g),key=lambda p:(-len(p),min(p)))
    for i,p in enumerate(ps):
        out=paint(out,shift(norm(p),i,0),colorof(g,p))
    return rotate(out,2)


def singleton_dock(g, anchor, outside):
    target=cells(g,anchor)
    if outside:
        target=rect(target,1,True)
    if not target:
        raise ValueError('anchor missing')
    movers=[p for p in components(g) if len(p)==1 and colorof(g,p)!=anchor]
    out=paint(g,set().union(*movers),modal(g))
    assignments={}
    for p in movers:
        r,c=next(iter(p))
        landing=min(target,key=lambda q:(abs(q[0]-r)+abs(q[1]-c),q))
        col=colorof(g,p)
        if landing in assignments and assignments[landing]!=col:
            raise ValueError('docking conflict')
        assignments[landing]=col
    return colored(out,assignments)


def farthest_corner_move(g):
    comps=components(g)
    frame=max(comps,key=len)
    movers=[p for p in comps if len(p)==1]
    if not movers:
        raise ValueError('no corner markers')
    t,l,b,r=bbox(frame)
    out=paint(g,set().union(*movers),modal(g))
    for cr,cc in ((t-1,l-1),(t-1,r+1),(b+1,l-1),(b+1,r+1)):
        selected=max(movers,key=lambda p:(abs(next(iter(p))[0]-cr)+abs(next(iter(p))[1]-cc),min(p)))
        out=paint(out,{(cr,cc)},colorof(g,selected))
    return out


def color_contact(g, mapping):
    out=g
    comps=components(g)
    movers=[p for p in comps if colorof(g,p) in dict(mapping)]
    out=paint(out,set().union(*movers),modal(g))
    for ps in movers:
        col=colorof(g,ps)
        target=cells(g,dict(mapping)[col])
        dr,dc=contact_delta(ps,target)
        out=paint(out,shift(ps,dr,dc),col)
    return out


def stemmed_block_contact(g):
    groups=[cells(g,c) for c in sorted(palette(g)-{modal(g)})]
    anchors=[p for p in groups if p==rect(p)]
    if len(anchors)!=1 or len(groups)!=2:
        raise ValueError('one block and one stemmed shape required')
    target=anchors[0]
    mover=next(p for p in groups if p!=target)
    interior={(r,c) for r,c in mover if all((r+dr,c+dc) in mover for dr,dc in ((1,1),(-1,1),(1,-1),(-1,-1)))}
    solid=rect(interior,1)
    dr,dc=contact_delta(solid,target)
    return paint(paint(g,mover,modal(g)),shift(solid,dr,dc),colorof(g,mover))


def lower_and_stripe(g, dst, step):
    comps=components(g)
    if len(comps)!=1:
        raise ValueError('one object required')
    ps=comps[0]
    moved=shift(ps,1,0)
    out=paint(paint(g,ps,modal(g)),moved,colorof(g,ps))
    top,left,_,_=bbox(moved)
    support={(r,c) for r in range(top) for c in range(len(g[0])) if (c-left)%step==0}
    return paint(out,support,dst)


def rotational_overlap_complete(g, src, dst, radius):
    ps=cells(g,src)
    q=cells(rotate(g),src)
    if not ps:
        raise ValueError('empty rotational support')
    options=[shift(q,dr,dc) for dr in range(-radius,radius+1) for dc in range(-radius,radius+1)]
    scores=[len(p&ps) for p in options]
    best=max(scores)
    selected=[p for p,s in zip(options,scores) if s==best]
    completions={paint(g,p,dst,under=True) for p in selected}
    if len(completions)!=1:
        raise ValueError('rotation alignment ambiguous')
    return completions.pop()


def symmetric_anchor_complete(g):
    ps=set().union(*components(g,False,True))
    anchors=[p for p in components(g,diag=True) if crop(g,p)==rotate(crop(g,p))]
    if len(anchors)!=1:
        raise ValueError('one rotationally symmetric anchor')
    cr,cc=center(anchors[0])
    assignment={}
    for r,c in ps:
        for rr,co in ((r,c),(2*cr-r,c),(r,2*cc-c),(2*cr-r,2*cc-c)):
            val=g[r][c]
            if (rr,co) in assignment and assignment[rr,co]!=val:
                raise ValueError('reflection conflict')
            assignment[rr,co]=val
    return colored(g,assignment)


def mirror_across_bar(g, mover, bar, gap):
    anchor=cells(g,bar)
    t,l,b,r=bbox(anchor)
    vertical=b-t>r-l
    x=g if vertical else rotate(g)
    comps=[p for p in components(x) if colorof(x,p)==mover]
    if len(comps)!=2:
        raise ValueError('two reflected movers required')
    comps.sort(key=lambda p:bbox(p)[0])
    out=paint(x,set().union(*comps),modal(x))
    for i,ps in enumerate(comps):
        top,left,bot,right=bbox(ps)
        delta=(bot-top+1)+gap
        moved={(top+bot-r+(delta if i==0 else -delta),c) for r,c in ps}
        out=paint(out,moved,mover)
    return out if vertical else rotate(out,3)


def candidates(demos):
    def propose():
        for fn in (height_rise,sorted_bars,farthest_corner_move,stemmed_block_contact,symmetric_anchor_complete):
            yield fn,()
        cols=colors(demos)
        for anchor in cols:
            for outside in (False,True):
                yield singleton_dock,(anchor,outside)
        for dst in added_colors(demos):
            yield lower_and_stripe,(dst,2)
            for src in cols:
                yield rotational_overlap_complete,(src,dst,2)
        # Candidate contacts are inferred per mover color on demonstrations.
        foreground=sorted(set.intersection(*(palette(a)-{modal(a)} for a,b in demos)))
        mappings=[]
        for mover in foreground:
            targets=[]
            for target in foreground:
                if mover==target:
                    continue
                ok=True
                for a,b in demos:
                    if shape(a)!=shape(b):
                        ok=False; break
                    try:
                        predicted=color_contact(a,((mover,target),))
                        if cells(predicted,mover)!=cells(b,mover):
                            ok=False;break
                    except ERRORS:
                        ok=False;break
                if ok:
                    targets.append(target)
            if targets:
                mappings.append([(mover,target) for target in targets])
        if mappings:
            for mapping in product(*mappings):
                yield color_contact,(tuple(mapping),)
        for mover,bar in permutations(foreground,2):
            yield mirror_across_bar,(mover,bar,3)
    return verified(demos,propose())
