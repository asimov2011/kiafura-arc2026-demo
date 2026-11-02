"""A5 neighborhood, ray, frontier and marker-connection graphs."""
from .a5_core import *


def neighborhood(ps,kind):
    ds=[(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and (kind=='all' or (abs(dr)+abs(dc)==1 if kind=='orthogonal' else dr!=0 and dc!=0))]
    return {(r+dr,c+dc) for r,c in ps for dr,dc in ds}


def neighbor_paint(g,rules):
    out=g
    for source,kind,mark in rules:out=fill(out,mark,neighborhood(cells(g,source),kind))
    return out


def diagonal_cross(g):
    objs=components(g)
    if len(objs)!=1:raise ValueError('one seed')
    ps=objs[0]; color=g[next(iter(ps))[0]][next(iter(ps))[1]]
    return fill(g,color,set().union(*(ray(g,center(ps),d) for d in [(-1,-1),(-1,1),(1,-1),(1,1)])))


def marked_cross(g,source,ring,centercolor):
    ps=cells(g,source);h,w=shape(g)
    lines={(r,c) for pr,pc in ps for r,c in cells(g) if r==pr or c==pc}
    out=fill(g,source,lines)
    out=fill(out,ring,neighborhood(ps,'diagonal'))
    return fill(out,centercolor,ps)


def component_cross_interior(g,background,raycolor,interior):
    objs=components(g,True,False,background)
    ps=set().union(*objs);out=g
    for o in objs:
        cr,cc=center(o)
        out=fill(out,raycolor,{(r,c) for r,c in cells(g) if r==cr or c==cc},under=background)
    return fill(out,interior,rectangle(ps)-rectangle(ps,edge=True))


def intersect_frontiers(g,horizontal,vertical,mark):
    hr=center(cells(g,horizontal))[0];vc=center(cells(g,vertical))[1]
    out=fill(g,horizontal,{(hr,c) for c in range(len(g[0]))})
    out=fill(out,vertical,{(r,vc) for r in range(len(g))})
    return fill(out,mark,{(hr,vc)})


def extend_oriented(g,background):
    out=g;objs=components(g,True,False,background)
    for horizontal in (False,True):
        for o in objs:
            a,b,x,y=bounds(o)
            if (a==x)!=horizontal:continue
            cr,cc=center(o);color=g[cr][cc]
            out=fill(out,color,{(cr,c) for c in range(len(g[0]))} if horizontal else {(r,cc) for r in range(len(g))})
    return out


def elbow_chain(g,firstcolor,middle,lastcolor,mark):
    ps=[cells(g,c) for c in (firstcolor,middle,lastcolor)]
    if any(len(p)!=1 for p in ps):raise ValueError('point roles')
    a,m,b=[next(iter(p)) for p in ps]
    bend1=(a[0],m[1]);bend2=(m[0],b[1])
    route=line(a,bend1)|line(bend1,m)|line(m,bend2)|line(bend2,b)
    return fill(g,mark,route,under=bg(g))


def missing_corner_triangle(g,marker,mask,mark):
    ps=cells(g,marker);a,b,x,y=bounds(ps)
    corners={(a,b),(a,y),(x,b),(x,y)};missing=corners-ps
    if len(missing)!=1:raise ValueError('three occupied corners')
    r,c=missing.pop();opposite=(x if r==a else a,y if c==b else b)
    adjacent=corners-{(r,c),opposite}
    u,v=sorted(adjacent)
    return fill(g,mark,(line((r,c),u)|line((r,c),v)|line(u,v))&cells(g,mask))


def mirror_intersection(g,mark):
    color=min(palette(g),key=lambda c:len(cells(g,c)))
    return fill(g,mark,cells(g,color)&cells(transform(g,4),color))


def alternating_diagonal(g,mark,background):
    out=g
    for o in components(g,True,True,background):
        a,b,_,_=bounds(o)
        out=fill(out,mark,{(a+k,b+k) for k in range(1,max(shape(g)),2)})
    return out


def outward_singleton_rays(g,background):
    objs=components(g,True,False,background);big=[o for o in objs if len(o)>1]
    if len(big)!=1:raise ValueError('one anchor')
    anchor=big[0]; ar,ac=center(anchor); color=g[next(iter(anchor))[0]][next(iter(anchor))[1]];out=g
    for o in objs:
        if len(o)==1:
            p=next(iter(o));d=(sign(p[0]-ar),sign(p[1]-ac))
            out=fill(out,color,ray(g,p,d))
    return out


def expanding_row(g,background):
    if len(g)!=1:raise ValueError('row seed')
    h=len(g[0])//2; out=g*h
    for color in palette(g):
        if color!=background:
            for p in cells(out,color):out=fill(out,color,ray(out,p,(1,1)))
    return out


def divider_rays(g,divider,inward,outward,background):
    split=min(r for r,c in cells(g,divider));out=g
    for p in cells(g,inward):
        d=1 if p[0]<split else -1
        route={q for q in ray(g,p,(d,0)) if (q[0]<split)==(p[0]<split)}
        out=fill(out,inward,route,under=background)
    for p in cells(g,outward):
        d=-1 if p[0]<split else 1
        out=fill(out,outward,ray(g,p,(d,0)))
    return out


def hanging_frames(g,objectcolor,shadow,frame,background):
    out=g
    for p in cells(g,objectcolor):out=fill(out,shadow,ray(g,p,(1,0)),under=background)
    for o in components(g,True,True,background):
        if not o&cells(g,objectcolor):continue
        a,b,x,y=bounds(o)
        for margin in range(1,(y-b+1)//2+1):out=fill(out,frame,rectangle(o,margin,True))
    return out


def stepped_frontiers(g,background):
    objs=sorted(components(g,True,False,background),key=lambda o:bounds(o)[0]);out=g
    h,w=shape(g)
    for j,o in enumerate(objs):
        r,c=center(o);color=g[next(iter(o))[0]][next(iter(o))[1]]
        out=fill(out,color,{(r,k) for k in range(c,w)})
        stop=bounds(objs[j+1])[0] if j+1<len(objs) else h
        out=fill(out,color,{(k,w-1) for k in range(r,stop)},under=background)
    return out


def candidates(demos):
    if not demos:return ()
    result=[];inc=colors(demos);outc=colors(demos,1)
    if not same_shape(demos):
        for b in inc:offer(result,demos,expanding_row,b)
        return tuple(result)
    rules=[]
    added=sorted({v for i,o in demos for r,c in cells(i) if i[r][c]!=o[r][c] for v in [o[r][c]]})
    for mark in added:
        opts=[]
        for source,kind in product(inc,('orthogonal','diagonal','all')):
            if all(cells(fill(i,mark,neighborhood(cells(i,source),kind)),mark)==cells(o,mark) for i,o in demos):opts.append((source,kind,mark))
        if opts:rules.append(opts)
    if rules and len(rules)<=3:
        for spec in product(*rules):offer(result,demos,neighbor_paint,spec)
    offer(result,demos,diagonal_cross)
    for source in inc:
        for ring,centercolor in permutations(outc,2):offer(result,demos,marked_cross,source,ring,centercolor)
    for background in inc:
        offer(result,demos,extend_oriented,background)
        offer(result,demos,outward_singleton_rays,background)
        offer(result,demos,stepped_frontiers,background)
        for mark in outc:offer(result,demos,alternating_diagonal,mark,background)
        for raycolor,interior in permutations(outc,2):offer(result,demos,component_cross_interior,background,raycolor,interior)
        for divider,inward,outward in permutations([c for c in inc if c!=background],3):offer(result,demos,divider_rays,divider,inward,outward,background)
        for objectcolor in inc:
            if objectcolor==background:continue
            for shadow,frame in permutations(outc,2):offer(result,demos,hanging_frames,objectcolor,shadow,frame,background)
    for mark in outc:
        offer(result,demos,mirror_intersection,mark)
        for a,b in permutations(inc,2):
            offer(result,demos,intersect_frontiers,a,b,mark)
            offer(result,demos,missing_corner_triangle,a,b,mark)
        singles=[c for c in inc if all(len(cells(i,c))==1 for i,o in demos)]
        for a,m,b in permutations(singles,3):offer(result,demos,elbow_chain,a,m,b,mark)
    return tuple(result)
