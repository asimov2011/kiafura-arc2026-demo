"""A4 nested rings, geometric slot assembly, cavity packing and motif matching."""
from itertools import product
from .a4_core import *


def fragmented_rings(g):
    bg=modal(g)
    groups=[cells(g,col) for col in sorted(palette(g)-{bg})]
    comps=components(g)
    def diameter(ps):
        same=[p for p in comps if colorof(g,p)==colorof(g,ps)]
        width=max(bbox(p)[3]-bbox(p)[1]+1 for p in same)
        distances=[min(abs(r-i)+abs(c-j) for r,c in a for i,j in b) for a in same for b in same if a!=b]
        return 2*width+(min(distances)-1 if distances else -2)
    groups.sort(key=lambda p:-diameter(p))
    depth=len(groups)+(0 if any(len(p)==1 for p in groups) else 1)
    n=2*depth-1
    out=canvas((n,n),bg)
    assignments={}
    for k,ps in enumerate(groups):
        q=norm(ps);t,l,b,r=bbox(q)
        variants=[{(i,r-j) for i,j in q},{(b-i,j) for i,j in q},
                  {(j,i) for i,j in q},{(r-j,b-i) for i,j in q}]
        q=min(variants,key=lambda p:(sum(i for i,j in p)//len(p),sum(j for i,j in p)//len(p)))
        for i,j in norm(q):
            assignments[i+k,j+k]=colorof(g,ps)
    for _ in range(4):
        out=colored(rotate(out),assignments)
    return out


def perimeter_slot_assembly(g,background,size):
    cleaned=paint(g,cells(g,background),0)
    comps=components(cleaned)
    center=size//2
    border={(r,c) for r in range(size) for c in range(size) if r in (0,size-1) or c in (0,size-1)}
    # One-cell separations between corner Ls and edge Ts.
    cuts={(r,c) for r,c in border if r in (2,size-3) or c in (2,size-3)}
    inner={(1,center),(size-2,center),(center,1),(center,size-2)}
    skeleton=paint(canvas((size,size)),(border-cuts)|inner,1)
    slots=components(skeleton)
    out=canvas((size,size),background)
    used=set()
    for ps in comps:
        matches=[p for p in slots if norm(p)==norm(ps)]
        if len(matches)!=1 or matches[0] in used:
            raise ValueError('slot assignment is not unique')
        used.add(matches[0])
        out=paint(out,matches[0],colorof(cleaned,ps))
    return out


def cavity_pack(g,mover,wall,dst,band_height):
    h,w=shape(g)
    if any(v!=wall for v in g[0]) or band_height>h:
        raise ValueError('top cavity wall required')
    allowed={(r,c) for r in range(band_height) for c in range(w)}
    empty=allowed&cells(g,0)
    blocked=allowed&cells(g,wall)
    movers=[p for p in components(g) if colorof(g,p)==mover]
    if not movers:
        raise ValueError('no movable pieces')
    if any(bbox(p)[0]<band_height for p in movers):
        raise ValueError('pieces must start below the cavity band')
    out=paint(g,set().union(*movers),modal(g))
    selected=[]
    for ps in movers:
        p=norm(ps)
        def score(q):
            return (10*len(q&allowed)-10*len(q&blocked)-len(rect(q,1,True)&empty)
                    -5*bbox(q)[0]-len((rect(q)-q)&empty))
        options=[shift(p,r,c) for r,c in sorted(allowed)]
        best=max(options,key=score)
        selected.append(best)
    return paint(out,set().union(*selected),dst)


def blank_motif_stamp(g,marker,blank):
    reference=cells(g,marker)
    ps=norm(reference)
    _,_,b,r=bbox(ps)
    h,w=b+1,r+1
    # Interior holes of a motif are negative space; they must remain nonblank.
    void=rect(ps)-ps
    external={p for p in void if p[0] in (0,h-1) or p[1] in (0,w-1)}
    todo=list(external)
    while todo:
        i,j=todo.pop()
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            p=i+di,j+dj
            if p in void and p not in external:
                external.add(p);todo.append(p)
    holes=void-external
    linear=h==1 or w==1
    matches=[]
    for top in range(len(g)-h+1):
        for left in range(len(g[0])-w+1):
            moved=shift(ps,top,left)
            if any(g[i][j]!=blank for i,j in moved):
                continue
            if any(g[i+top][j+left]==blank for i,j in holes):
                continue
            if linear:
                ends=(min(moved),max(moved))
                sides=((1,0),(-1,0)) if h==1 else ((0,1),(0,-1))
                if any(0<=i+di<len(g) and 0<=j+dj<len(g[0]) and g[i+di][j+dj]==blank
                       for i,j in ends for di,dj in sides):
                    continue
            perimeter={(i+di,j+dj) for i,j in moved for di,dj in ((1,0),(-1,0),(0,1),(0,-1))}-moved
            exposure=sum(0<=i<len(g) and 0<=j<len(g[0]) and g[i][j]==blank for i,j in perimeter)
            matches.append((exposure,top,left,moved))
    support=set()
    for index,(exposure,top,left,moved) in enumerate(sorted(matches)):
        if support&moved:
            continue
        if any(other_score==exposure and other!=moved and other&moved
               for other_score,_,_,other in matches):
            raise ValueError('equally exposed overlapping motifs')
        support|=moved
    return paint(g,support,marker)


def candidates(demos):
    def propose():
        yield fragmented_rings,()
        a,b=demos[0]
        cols=colors(demos)
        if len(b)==len(b[0]):
            for background in cols:
                yield perimeter_slot_assembly,(background,len(b))
        # Packing region is observed through cells whose color changes in demos.
        if all(shape(x)==shape(y) for x,y in demos):
            for dst in added_colors(demos):
                changed=[r for x,y in demos for r,row in enumerate(y) for c,v in enumerate(row) if v==dst and x[r][c]!=dst]
                if not changed:
                    continue
                bh=max(changed)+1
                for mover,wall in product(cols,repeat=2):
                    if mover!=wall:
                        yield cavity_pack,(mover,wall,dst,bh)
        for marker,blank in product(cols,repeat=2):
            if marker!=blank:
                yield blank_motif_stamp,(marker,blank)
    return verified(demos,propose())
