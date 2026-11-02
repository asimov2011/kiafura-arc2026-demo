"""A5 panel logic, selection, rotational completion and marker-bound layout."""
from .a5_core import *


def halves(g,axis):
    if not g or not g[0]:return (),()
    if axis==0:
        n=len(g)//2
        return g[:n],g[-n:]
    n=len(g[0])//2
    return tuple(tuple(r[:n]) for r in g),tuple(tuple(r[-n:]) for r in g)


def quads(g):
    top,bot=halves(g,0)
    return halves(top,1)+halves(bot,1)


def panel_logic(g,axis,mapping):
    a,b=halves(g,axis); table=dict(mapping)
    return tuple(tuple(table[(x,y)] for x,y in zip(ar,br)) for ar,br in zip(a,b))


def quadrant_overlay(g,background,order):
    qs=quads(g); out=qs[order[0]]
    for k in order[1:]:
        out=paste(out,qs[k],cells(qs[k])-cells(qs[k],background))
    return out


def colorful_quadrant(g):
    qs=quads(g); n=max(len(palette(q)) for q in qs)
    best=[q for q in qs if len(palette(q))==n]
    if len(best)!=1: raise ValueError('ambiguous panel')
    return best[0]


def missing_rotation(g):
    qs=quads(g)
    out={transform(qs[0],k) for k in range(4)}-set(qs[:3])
    if len(out)!=1: raise ValueError('rotation not unique')
    return out.pop()


def panel_rotation_sequence(g,n,gap,ks):
    h,w=shape(g)
    if h!=n or w!=(n+gap)*len(ks)-gap: raise ValueError('panel layout')
    first=tuple(row[:n] for row in g)
    out=g
    for index,k in enumerate(ks):
        tile=transform(first,k)
        out=paste(out,tile,cells(tile),(0,index*(n+gap)))
    return out


def marker_roll(g,marker):
    h,w=shape(g); ps=cells(g,marker)
    if len(ps)!=1 or next(iter(ps))[1]!=0: raise ValueError('side marker')
    r=next(iter(ps))[0]; core=tuple(row[1:] for row in g)
    return core[-r:]+core[:-r] if r else core


def framed_repeat(g,k,edge,corner,background):
    core=transform(g,k); h,w=shape(core)
    out=canvas(background,(h+2,w+2))
    out=fill(out,edge,rectangle(cells(out),edge=True))
    out=fill(out,corner,{(0,0),(0,w+1),(h+1,0),(h+1,w+1)})
    out=paste(out,core,cells(core),(1,1))
    return out+out[1:]


def motif_counter(g,motif,counter,n,background):
    ps=cells(g,motif); a,b,x,y=bounds(ps)
    tile=fill(canvas(background,(n,n)),motif,move(norm(ps),(n-(x-a+1),n-(y-b+1))))
    count=len(cells(g,counter))
    if not 1<=count<=10: raise ValueError('count')
    return hcat([tile]*count)


def quadrant_summaries(g,frame,background):
    qs=quads(crop(g,cells(g,frame)))
    vals=[]
    for q in qs:
        color=min(palette(q),key=lambda c:len(cells(q,c)))
        vals.append(background if color==frame else color)
    return (tuple(vals[:2]),tuple(vals[2:]))


def candidates(demos):
    if not demos:return ()
    result=[]
    for axis in (0,1):
        if all(shape(halves(i,axis)[0])==shape(o) for i,o in demos):
            mapping={}; valid=True
            for i,o in demos:
                a,b=halves(i,axis)
                for ar,br,orr in zip(a,b,o):
                    for x,y,z in zip(ar,br,orr):
                        if (x,y) in mapping and mapping[x,y]!=z: valid=False
                        mapping[x,y]=z
            if valid: offer(result,demos,panel_logic,axis,tuple(sorted(mapping.items())))
    if all(shape(quads(i)[0])==shape(o) for i,o in demos):
        offer(result,demos,colorful_quadrant)
        offer(result,demos,missing_rotation)
        for background in colors(demos):
            for order in permutations(range(4)): offer(result,demos,quadrant_overlay,background,order)
    for marker in colors(demos): offer(result,demos,marker_roll,marker)
    h,w=shape(demos[0][0]); ho,wo=shape(demos[0][1])
    if all(shape(i)==shape(o) for i,o in demos) and w>h:
        for gap in (0,1):
            if (w+gap)%(h+gap):continue
            count=(w+gap)//(h+gap)
            if not 2<=count<=4:continue
            choices=[]
            for j in range(count):
                choices.append([k for k in range(4) if all(transform(tuple(row[:len(i)] for row in i),k)==tuple(row[j*(len(i)+gap):j*(len(i)+gap)+len(i)] for row in o) for i,o in demos)])
            for ks in product(*choices): offer(result,demos,panel_rotation_sequence,h,gap,ks)
    if ho in (2*(h+2)-1,2*(w+2)-1):
        for k in range(4):
            for edge,corner in permutations(colors(demos,1),2):
                offer(result,demos,framed_repeat,k,edge,corner,0)
    if ho<=5:
        for motif,counter in permutations(colors(demos),2):
            for background in colors(demos,1):offer(result,demos,motif_counter,motif,counter,ho,background)
    if (ho,wo)==(2,2):
        for frame in colors(demos):
            for background in colors(demos,1):offer(result,demos,quadrant_summaries,frame,background)
    return tuple(result)
