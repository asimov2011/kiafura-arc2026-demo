"""A5 input-relative translation, reflection, attraction and marker navigation."""
from .a5_core import *


def shift_foreground(g,background,outbackground,delta):
    return paste(canvas(outbackground,shape(g)),g,cells(g)-cells(g,background),delta)


def axis_marker_move(g,marker,moving,background,step):
    a,b,x,y=bounds(cells(g,marker)); delta=(0,step) if a==x else (step,0)
    ps=cells(g,moving)
    return fill(fill(g,background,ps),moving,move(ps,delta))


def point_reflection(g,marker,background):
    ps=cells(g,marker)
    if len(ps)!=1:raise ValueError('unique center')
    pr,pc=next(iter(ps));h,w=shape(g)
    return tuple(tuple(g[2*pr-r][2*pc-c] if 0<=2*pr-r<h and 0<=2*pc-c<w else background for c in range(w)) for r in range(h))


def center_constellation(g,background,mark):
    objs=components(g,True,False,background);pr,pc=center(cells(g)-cells(g,background));out=g
    if len(objs)<2:raise ValueError('constellation')
    for o in objs:
        r,c=center(o)
        out=paste(out,g,o,(pr+sign(r-pr)-r,pc+sign(c-pc)-c))
    return fill(out,mark,{(pr,pc)})


def singleton_attraction(g,background):
    objs=components(g,True,False,background);big=[o for o in objs if len(o)>1];small=[o for o in objs if len(o)==1]
    out=fill(g,background,set().union(*small))
    for o in small:
        r,c=next(iter(o)); color=g[r][c];targets=[b for b in big if b&cells(g,color)]
        if len(targets)!=1:continue
        target=targets[0]; options=[]
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            for n in range(max(shape(g))):
                p=r+n*dr,c+n*dc
                if p in target:break
                if any(abs(p[0]-tr)+abs(p[1]-tc)==1 for tr,tc in target):
                    options.append((n,p));break
        if not options:raise ValueError('unreachable target')
        distance=min(n for n,p in options);best={p for n,p in options if n==distance}
        if len(best)!=1:raise ValueError('ambiguous attraction')
        out=fill(out,color,best)
    return out


def point_toward_target(g,moving,target,trail):
    a=cells(g,moving);b=cells(g,target)
    if len(a)!=1 or len(b)!=1:raise ValueError('two points')
    r,c=next(iter(a));tr,tc=next(iter(b))
    q=(r+(sign(tr-r) if abs(tr-r)>1 else 0),c+(sign(tc-c) if abs(tc-c)>1 else 0))
    return fill(fill(fill(g,trail,a),moving,{q}),moving,b)


def obstacle_corner_step(g,obstacle,moving,target,trail):
    a,b,x,y=bounds(cells(g,obstacle));ps=cells(g,moving);qs=cells(g,target)
    if len(ps)!=1 or len(qs)!=1:raise ValueError('point roles')
    r,c=next(iter(ps)); q=(x,y+1 if c<(b+y)//2 else b-1)
    out=fill(g,trail,ps);out=fill(out,moving,qs)
    return fill(out,target,{q})


def candidates(demos):
    if not demos:return ()
    result=[]
    if not same_shape(demos):return ()
    inc=colors(demos);outc=colors(demos,1)
    for background in inc:
        offer(result,demos,singleton_attraction,background)
        for mark in outc:offer(result,demos,center_constellation,background,mark)
        for marker in inc:offer(result,demos,point_reflection,marker,background)
        for outbackground in outc:
            for dr,dc in [(0,-1),(0,1),(-1,0),(1,0)]:offer(result,demos,shift_foreground,background,outbackground,(dr,dc))
        for marker,moving in permutations([c for c in inc if c!=background],2):
            for step in (-2,-1,1,2):offer(result,demos,axis_marker_move,marker,moving,background,step)
    singles=[c for c in inc if all(len(cells(i,c))==1 for i,o in demos)]
    for moving,target in permutations(singles,2):
        for trail in outc:
            offer(result,demos,point_toward_target,moving,target,trail)
            for obstacle in inc:
                if obstacle not in (moving,target):offer(result,demos,obstacle_corner_step,obstacle,moving,target,trail)
    return tuple(result)
