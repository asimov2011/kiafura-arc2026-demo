"""A5 periodic continuation, symmetry repair, framed motion and spatial fields."""
from .a5_core import *


def concentric_repair(g,background):
    h,w=shape(g)
    if h!=w:raise ValueError('square field')
    out=g
    for d in range((h+1)//2):
        ring={(r,c) for r,c in cells(g) if min(r,c,h-1-r,w-1-c)==d}
        cs={g[r][c] for r,c in ring}-{background}
        if len(cs)>1:raise ValueError('ring color conflict')
        if cs:out=fill(out,cs.pop(),ring)
    return out


def symmetry_repair(g,occluder,ks):
    views=[transform(g,k) for k in ks]
    if any(shape(q)!=shape(g) for q in views):raise ValueError('symmetry shape')
    out=[]
    for r,row in enumerate(g):
        dest=[]
        for c,v in enumerate(row):
            cs={q[r][c] for q in views}-{occluder}
            if len(cs)!=1:raise ValueError('symmetry unresolved')
            dest.append(cs.pop())
        out.append(tuple(dest))
    return tuple(out)


def shifted_periodic_field(g,mask):
    if mask=='corner':mask=g[-1][-1]
    # The cohort's operation moves the visible periodic field away from its
    # terminal masked columns. Direction is structural, not an ambiguous fit.
    right=all(row[-1]==mask for row in g);left=all(row[0]==mask for row in g)
    if right==left:raise ValueError('one terminal masked side')
    shift=1 if right else -1
    h,w=shape(g);known=cells(g)-cells(g,mask)
    candidates=[]
    for ph in range(1,h+1):
        for pw in range(1,w+1):
            if ph*pw>len(known)//2:continue
            tile={};valid=True
            for r,c in sorted(known):
                p=(r%ph,c%pw);color=g[r][c]
                if p in tile and tile[p]!=color:valid=False;break
                tile[p]=color
            if valid and len(tile)==ph*pw:candidates.append((ph*pw,ph,pw,tile))
    if not candidates:raise ValueError('no repeated period')
    area=min(q[0] for q in candidates);outs=set()
    for _,ph,pw,tile in candidates:
        if ph*pw==area:outs.add(tuple(tuple(tile[r%ph,(c+shift)%pw] for c in range(w)) for r in range(h)))
    if len(outs)!=1:raise ValueError('period ambiguity')
    return outs.pop()


def sparse_square_continuation(g,background):
    objs=components(g,True,True,background)
    if len(objs)!=1:raise ValueError('one periodic object')
    ps=objs[0];a,b,x,y=bounds(ps);h,w=x-a+1,y-b+1;out=canvas(background,(len(g[0]),len(g[0])))
    out=paste(out,g,ps)
    if h>w:
        period=h
        for p in range(1,h+1):
            if all((r+p,c) in ps and g[r+p][c]==g[r][c] for r,c in ps if r+p<=x):period=p;break
        for dr in range(period,len(out),period):out=paste(out,g,ps,(dr,0))
    else:out=paste(out,g,ps,(h,w-1))
    return out


def corner_block_extension(g,background):
    h,w=shape(g)
    if h!=w:raise ValueError('square seed')
    core=tuple(row[:-1] for row in g[:-1]) if background in palette(g) else g
    n=len(core);row=core[0];top=hcat([core,tuple(row for _ in range(n))])
    lower=hcat([tuple((c,)*n for c in row),core])
    q=top+lower
    return paste(canvas(g[0][0],(h*2,w*2)),q,cells(q))


def bracket_ejection(g,frame,moving,background):
    parts=[o for o in components(g,True,False,background) if o&cells(g,frame)]
    if len(parts)!=2:raise ValueError('paired brackets')
    a,b,x,y=bounds(cells(g,frame));o=parts[0];u,v,z,t=bounds(o);vertical=z-u>t-v
    core=tuple(tuple(row[b+1:y]) for row in g[a+1:x]);out=fill(g,background,cells(g,moving))
    if vertical:
        n=len(core[0])//2
        left=transform(tuple(row[:n] for row in core),4);right=transform(tuple(row[-n:] for row in core),4)
        out=paste(out,left,cells(left),(a+1,b-n))
        out=paste(out,right,cells(right),(a+1,b+2*n+2))
    else:
        n=len(core)//2
        top=core[:n][::-1];bottom=core[-n:][::-1]
        out=paste(out,top,cells(top),(a-n,b+1))
        out=paste(out,bottom,cells(bottom),(a+2*n+2,b+1))
    return out


def opening_fan(g,background,mark):
    ps=cells(g)-cells(g,background);box=rectangle(ps,edge=True);opening=box-ps
    if not opening:raise ValueError('no opening')
    a,b,x,y=bounds(ps);u,v,z,t=bounds(opening)
    if u==z and u in (a,x):d=(-1 if u==a else 1,0);ends=[(u,v),(u,t)];diags=[(d[0],-1),(d[0],1)]
    elif v==t and v in (b,y):d=(0,-1 if v==b else 1);ends=[(u,v),(z,v)];diags=[(-1,d[1]),(1,d[1])]
    else:raise ValueError('single side opening')
    out=fill(g,mark,rectangle(ps)-ps)
    for p in opening:out=fill(out,mark,ray(g,p,d))
    for p,delta in zip(ends,diags):out=fill(out,mark,ray(g,p,delta))
    return out


def reflected_beam(g,background,mark):
    objs=components(g,True,True,background)
    if len(objs)!=2:raise ValueError('beam and obstacle')
    seed=min(objs,key=len);wall=max(objs,key=len);a,b,x,y=bounds(seed)
    if x==a or y==b:raise ValueError('diagonal beam seed')
    p=min(seed);other=max(seed);d=(sign(other[0]-p[0]),sign(other[1]-p[1]))
    first=(ray(g,p,d)|ray(g,p,(-d[0],-d[1])))&cells(g)-wall
    adjacent={p for p in first if any(abs(p[0]-r)+abs(p[1]-c)==1 for r,c in wall)}
    if len(adjacent)!=1:raise ValueError('one bounce point')
    hit=adjacent.pop();second=ray(g,hit,(d[0],-d[1]))|ray(g,hit,(-d[0],d[1]))
    out=fill(g,mark,first|second)
    return paste(out,g,seed|wall)


def blank_corridor_interior(g,background,mark):
    """Select maximal empty rectangle bands, then their joint interior.

    Connected-band closure removes the largest remaining rectangular band;
    termination follows from decreasing available cells, not a fitted pass count.
    No coordinates,
    source programs or expected grids are retained. Boundary corridors stay open.
    """
    h,w=shape(g);available=cells(g,background);mask=set()
    while available:
        best=0;bands=[]
        for a in range(h):
            cols=set(range(w))
            for x in range(a,h):
                cols&={c for c in range(w) if (x,c) in available}
                if x-a+1<3:continue
                runs=[];start=None
                for c in range(w+1):
                    if c in cols:
                        if start is None:start=c
                    elif start is not None:
                        if c-start>=3:runs.append((start,c-1))
                        start=None
                for b,y in runs:
                    if mask and not any((r,b-1) in mask or (r,y+1) in mask for r in range(a,x+1)) and not any((a-1,c) in mask or (x+1,c) in mask for c in range(b,y+1)):
                        continue
                    area=(x-a+1)*(y-b+1)
                    if area>best:best=area;bands=[]
                    if area==best:bands.append((a,b,x,y))
        if not bands:break
        chosen=set().union(*({(r,c) for r in range(a,x+1) for c in range(b,y+1)} for a,b,x,y in bands))
        mask|=chosen;available-=chosen
    inside={p for p in cells(g) if all((p[0]+dr,p[1]+dc) in mask for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc)}
    border={p for p in cells(g,background) if p[0] in (0,h-1) or p[1] in (0,w-1)}
    edges={p for p in border if any((p[0]+dr,p[1]+dc) in inside for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)])}
    return fill(g,mark,inside|edges)


def candidates(demos):
    if not demos:return ()
    result=[];inc=colors(demos);outc=colors(demos,1)
    if same_shape(demos):
        offer(result,demos,shifted_periodic_field,'corner')
        for background in inc:
            offer(result,demos,concentric_repair,background)
            offer(result,demos,shifted_periodic_field,background)
            for ks in [(0,4),(0,6),(0,2,5,7),(0,2,4,6),tuple(range(8))]:offer(result,demos,symmetry_repair,background,ks)
            for mark in outc:
                offer(result,demos,opening_fan,background,mark)
                offer(result,demos,reflected_beam,background,mark)
                if min(shape(demos[0][0]))>=15 and mark not in palette(demos[0][0]):
                    offer(result,demos,blank_corridor_interior,background,mark)
            for frame,moving in permutations([c for c in inc if c!=background],2):offer(result,demos,bracket_ejection,frame,moving,background)
    else:
        for background in inc:
            offer(result,demos,sparse_square_continuation,background)
        for background in sorted({c for i,o in demos for c in palette(i)}):
            offer(result,demos,corner_block_extension,background)
    return tuple(result)
