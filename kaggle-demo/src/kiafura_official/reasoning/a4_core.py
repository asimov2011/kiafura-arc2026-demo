"""A4 grid primitives and demo verification. No learned external dependencies."""
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache


def grid(a):
    g = tuple(tuple(row) for row in a)
    if not g or not g[0] or len(g) > 60 or len(g[0]) > 60:
        raise ValueError('invalid extent')
    if any(len(row) != len(g[0]) for row in g):
        raise ValueError('ragged grid')
    return g


def shape(g):
    return len(g), len(g[0])


def palette(g):
    return set(v for row in g for v in row)


def modal(g):
    ct = Counter(v for row in g for v in row)
    return min(ct, key=lambda x: (-ct[x], x))


def rare(g):
    ct = Counter(v for row in g for v in row)
    return min(ct, key=lambda x: (ct[x], x))


def cells(g, color):
    return {(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v == color}


def bbox(ps):
    if not ps:
        raise ValueError('empty support')
    return min(r for r,c in ps), min(c for r,c in ps), max(r for r,c in ps), max(c for r,c in ps)


def rect(ps, margin=0, border=False):
    t,l,b,r = bbox(ps)
    t,l,b,r = t-margin,l-margin,b+margin,r+margin
    return {(i,j) for i in range(t,b+1) for j in range(l,r+1)
            if not border or i in (t,b) or j in (l,r)}


def norm(ps):
    t,l,_,_ = bbox(ps)
    return frozenset((r-t,c-l) for r,c in ps)


def center(ps):
    t,l,b,r = bbox(ps)
    return t+(b-t+1)//2, l+(r-l+1)//2


def crop(g, ps):
    t,l,b,r = bbox(ps)
    if t < 0 or l < 0 or b >= len(g) or r >= len(g[0]):
        raise ValueError('crop outside grid')
    return grid(row[l:r+1] for row in g[t:b+1])


def canvas(s, color=0):
    h,w = s
    if not 0 < h <= 60 or not 0 < w <= 60:
        raise ValueError('canvas extent')
    return tuple((color,)*w for _ in range(h))


def paint(g, ps, color, under=False):
    out = [list(row) for row in g]
    bg = modal(g)
    for r,c in ps:
        if 0 <= r < len(g) and 0 <= c < len(g[0]) and (not under or g[r][c] == bg):
            out[r][c] = color
    return grid(out)


def colored(g, assignments):
    out = [list(row) for row in g]
    for (r,c),v in assignments.items():
        if 0 <= r < len(g) and 0 <= c < len(g[0]):
            out[r][c] = v
    return grid(out)


def shift(ps, dr, dc):
    return {(r+dr,c+dc) for r,c in ps}


def rotate(g, n=1):
    for _ in range(n % 4):
        g = tuple(zip(*g[::-1]))
    return g


def transpose(g):
    return tuple(zip(*g))


def scale(g, rh, cw=None):
    cw = rh if cw is None else cw
    if len(g)*rh > 60 or len(g[0])*cw > 60 or min(rh,cw) < 1:
        raise ValueError('scale extent')
    return tuple(tuple(v for x in row for v in (x,)*cw) for row in g for _ in range(rh))


@lru_cache(maxsize=1024)
def components(g, mono=True, diag=False, include_bg=False):
    bg = modal(g)
    remaining = {(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if include_bg or v != bg}
    result = []
    dirs = [(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and (diag or not dr or not dc)]
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        found = {seed}
        todo = deque([seed])
        col = g[seed[0]][seed[1]]
        while todo:
            r,c = todo.popleft()
            for dr,dc in dirs:
                p = r+dr,c+dc
                if p in remaining and (not mono or g[p[0]][p[1]] == col):
                    remaining.remove(p)
                    found.add(p)
                    todo.append(p)
        result.append(frozenset(found))
    return tuple(result)


def colorof(g, ps):
    cols = {g[r][c] for r,c in ps}
    if len(cols) != 1:
        raise ValueError('not monochrome')
    return next(iter(cols))


def line(a,b):
    dr,dc = b[0]-a[0],b[1]-a[1]
    if dr and dc and abs(dr) != abs(dc):
        return set()
    n = max(abs(dr),abs(dc))
    if not n:
        return {a}
    return {(a[0]+i*dr//n,a[1]+i*dc//n) for i in range(n+1)}


def ray(g, p, d):
    if d == (0,0):
        raise ValueError('zero direction')
    if d[0] and d[1] and abs(d[0])!=abs(d[1]):
        return set()
    d=tuple((v>0)-(v<0) for v in d)
    return {(p[0]+i*d[0],p[1]+i*d[1]) for i in range(2*max(shape(g)))
            if 0 <= p[0]+i*d[0] < len(g) and 0 <= p[1]+i*d[1] < len(g[0])}


def split_panels(g, axis, n):
    x = transpose(g) if axis else g
    h = len(x)
    size, sep = h//n, 0
    if h % n == n-1:
        size,sep = (h-(n-1))//n,1
    elif h % n:
        raise ValueError('unequal panels')
    parts = [grid(x[i*(size+sep):i*(size+sep)+size]) for i in range(n)]
    return [transpose(x) for x in parts] if axis else parts


ERRORS = (ValueError, IndexError, KeyError, StopIteration, ZeroDivisionError)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple
    operation: object

    def execute(self, source):
        return grid(self.operation(source, *self.bindings))


def verified(demos, proposals):
    if not demos:
        return ()
    result = []
    seen = set()
    for fn, bindings in proposals:
        key = fn.__name__, bindings
        if key in seen:
            continue
        seen.add(key)
        candidate = Candidate(fn.__name__, bindings, fn)
        try:
            if all(candidate.execute(a) == b for a,b in demos):
                result.append(candidate)
        except ERRORS:
            continue
    return tuple(result)


def colors(demos, output=False):
    return sorted(set.union(*(palette(b if output else a) for a,b in demos)))


def added_colors(demos):
    return sorted(set.union(*(palette(b)-palette(a) for a,b in demos))) or colors(demos, True)
