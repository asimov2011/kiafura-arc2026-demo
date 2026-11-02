"""A5 grid primitives and exact-demo filtering; no source-program machinery."""
from collections import Counter
from dataclasses import dataclass
from itertools import product, permutations

ERRORS = (ValueError, IndexError, KeyError, StopIteration, ZeroDivisionError, TypeError)


def grid(g):
    return tuple(tuple(row) for row in g)


def shape(g):
    return len(g), len(g[0]) if g else 0


def palette(g):
    return sorted({c for row in g for c in row})


def bg(g):
    return Counter(c for row in g for c in row).most_common(1)[0][0]


def cells(g, color=None):
    return {(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if color is None or v==color}


def bounds(ps):
    if not ps:
        raise ValueError('empty region')
    return min(r for r,c in ps), min(c for r,c in ps), max(r for r,c in ps), max(c for r,c in ps)


def rectangle(ps, margin=0, edge=False):
    a,b,x,y=bounds(ps)
    a,b,x,y=a-margin,b-margin,x+margin,y+margin
    return {(r,c) for r in range(a,x+1) for c in range(b,y+1) if not edge or r in (a,x) or c in (b,y)}


def center(ps):
    a,b,x,y=bounds(ps)
    return (a+x)//2,(b+y)//2


def move(ps, delta):
    dr,dc=delta
    return {(r+dr,c+dc) for r,c in ps}


def norm(ps):
    a,b,_,_=bounds(ps)
    return move(ps,(-a,-b))


def components(g, mono=True, diagonal=False, background=None, include_bg=False):
    background=bg(g) if background is None else background
    todo={p for p in cells(g) if include_bg or g[p[0]][p[1]]!=background}
    ds=[(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and (diagonal or abs(dr)+abs(dc)==1)]
    result=[]
    while todo:
        seed=min(todo); todo.remove(seed); stack=[seed]; region={seed}
        color=g[seed[0]][seed[1]]
        while stack:
            r,c=stack.pop()
            for dr,dc in ds:
                p=r+dr,c+dc
                if p in todo and (not mono or g[p[0]][p[1]]==color):
                    todo.remove(p); region.add(p); stack.append(p)
        result.append(region)
    return result


def crop(g, ps):
    a,b,x,y=bounds(ps)
    return tuple(tuple(g[r][c] for c in range(b,y+1)) for r in range(a,x+1))


def canvas(color, dims):
    h,w=dims
    if not (0<h<=60 and 0<w<=60):
        raise ValueError('invalid canvas')
    return tuple((color,)*w for _ in range(h))


def fill(g,color,ps,under=None):
    out=[list(row) for row in g]; h,w=shape(g)
    for r,c in ps:
        if 0<=r<h and 0<=c<w and (under is None or out[r][c]==under):
            out[r][c]=color
    return grid(out)


def paste(g, source, ps, delta=(0,0), under=None):
    out=g
    for color in palette(source):
        out=fill(out,color,move(ps & cells(source,color),delta),under)
    return out


def transform(g, k):
    if k>=4:
        g=tuple(tuple(reversed(row)) for row in g); k-=4
    for _ in range(k):
        g=tuple(zip(*g[::-1]))
    return grid(g)


def scale(g,n):
    return tuple(tuple(v for c in row for v in [c]*n) for row in g for _ in range(n))


def hcat(gs):
    return tuple(sum(rows,()) for rows in zip(*gs))


def sign(x):
    return (x>0)-(x<0)


def line(a,b):
    dr,dc=b[0]-a[0],b[1]-a[1]
    if dr and dc and abs(dr)!=abs(dc):
        return set()
    return {(a[0]+i*sign(dr),a[1]+i*sign(dc)) for i in range(max(abs(dr),abs(dc))+1)}


def ray(g,p,d):
    h,w=shape(g)
    return {(p[0]+i*d[0],p[1]+i*d[1]) for i in range(max(h,w)*2+1)}


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple
    fn: object

    def execute(self,g):
        result=grid(self.fn(g,*self.bindings))
        h,w=shape(result)
        if not 0<h<=30 or not 0<w<=30 or any(len(row)!=w or any(not isinstance(c,int) or not 0<=c<=9 for c in row) for row in result):
            raise ValueError('invalid ARC output')
        return result


def exact(demos, graph, fn, bindings=()):
    c=Candidate(graph,tuple(bindings),fn)
    try:
        if demos and all(c.execute(i)==o for i,o in demos):
            return c
    except ERRORS:
        pass
    return None


def offer(result,demos,fn,*bindings):
    c=exact(demos,fn.__name__,fn,bindings)
    if c is not None:
        result.append(c)


def colors(demos,which=0):
    return sorted(set.intersection(*(set(palette(p[which])) for p in demos)))


def same_shape(demos):
    return all(shape(i)==shape(o) for i,o in demos)
