"""A4 templates are extracted anew from each input, never stored as grids."""
from itertools import product
from .a4_core import *
from .a4_pattern_algebra import windows


def corner_stamp(g, size, marker):
    template=grid(row[:size] for row in g[:size])
    assignments={}
    for r,c in cells(g,marker):
        for i,row in enumerate(template):
            for j,v in enumerate(row):
                assignments[r+i-size//2,c+j-size//2]=v
    return colored(g,assignments)


def corner_shape_recolor(g, size, dst):
    template=grid(row[:size] for row in g[:size])
    bg=modal(g)
    support=norm({(r,c) for r,row in enumerate(template) for c,v in enumerate(row) if v!=bg})
    out=g
    for ps in components(g):
        if norm(ps)==support:
            out=paint(out,ps,dst)
    return colored(out,{(i,j):v for i,row in enumerate(template) for j,v in enumerate(row)})


def partial_shape_complete(g, dst):
    comps=sorted(components(g,diag=True),key=lambda p:bbox(p)[0])
    if len(comps)<2:
        raise ValueError('template and fragments required')
    reference=norm(comps[0])
    out=g
    for ps in comps[1:]:
        t,l,b,r=bbox(ps)
        options=[shift(reference,t,l+offset) for offset in range(3)]
        best=max(options,key=lambda p:(len(p&ps),bbox(p)[1]))
        out=paint(out,best,dst,under=True)
    return out


def row_palette_template(g):
    comps=components(g,False,True)
    if len(comps)<2:
        raise ValueError('template and palettes required')
    template=max(comps,key=len)
    t,l,b,r=bbox(template)
    if t!=b:
        raise ValueError('horizontal template required')
    pattern=tuple(g[t][c] for c in range(l,r+1))
    source_order=list(dict.fromkeys(pattern))
    if len(source_order)!=2:
        raise ValueError('two roles required')
    out=g
    for ps in comps:
        if ps==template:
            continue
        top,left,bot,right=bbox(ps)
        target_order=list(dict.fromkeys(g[top][c] for c in range(left,right+1)))
        if top!=bot or len(target_order)!=2:
            raise ValueError('two-color palette strip')
        mapping=dict(zip(source_order,target_order))
        out=colored(out,{(top,c):mapping[v] for c,v in enumerate(pattern)})
    return out


def horizontal_period_complete(g):
    bg=modal(g)
    comps=components(g,False)
    if not comps:
        raise ValueError('no periodic object')
    ps=max(comps,key=len)
    t,l,b,r=bbox(ps)
    observed={(i,j):g[i][j] for i,j in ps}
    width=r-l+1
    period=None
    for p in range(1,width):
        if all((i,j-p) in observed and observed[i,j-p]==v for (i,j),v in observed.items() if j>=l+p):
            period=p;break
    if period is None:
        raise ValueError('no period witness')
    out=g
    for c in range(l,len(g[0])):
        origin=l+(c-l)%period
        for i in range(t,b+1):
            if (i,origin) in observed:
                out=paint(out,{(i,c)},observed[i,origin])
    return out


def scaled_diagonal_echo(g, dst, factor):
    support=set()
    for ps in components(g,diag=True):
        t,l,b,r=bbox(ps)
        h,w=b-t+1,r-l+1
        reflected={(i,l+r-j) for i,j in ps}
        expanded={(t+(i-t)*factor+di-h//2,l+(j-l)*factor+dj-w//2)
                  for i,j in reflected for di in range(factor) for dj in range(factor)}
        rows={i for i,j in ps}; cols={j for i,j in ps}
        support|={(i,j) for i,j in expanded if i not in rows and j not in cols}
    return paint(g,support,dst)


def lattice_template_complete(g):
    rr,cc=windows(g)
    reduced=tuple(tuple(g[rs[0]][cs[0]] for cs in cc) for rs in rr)
    bg=0
    candidates=[]
    for col in sorted(palette(reduced)-{bg}):
        marks=cells(reduced,col)
        for r,c in marks:
            context={(dr,dc):reduced[r+dr][c+dc] for dr in (-1,0,1) for dc in (-1,0,1)
                     if 0<=r+dr<len(rr) and 0<=c+dc<len(cc)}
            other=sum(v not in (bg,col) for v in context.values())
            if other:
                candidates.append((other,col,r,c,context))
    if not candidates:
        raise ValueError('no populated lattice template')
    _,col,tr,tc,context=max(candidates,key=lambda x:x[:4])
    out=g
    for r,c in cells(reduced,col):
        for (dr,dc),v in context.items():
            if 0<=r+dr<len(rr) and 0<=c+dc<len(cc):
                out=paint(out,{(i,j) for i in rr[r+dr] for j in cc[c+dc]},v)
    return out


def scaled_marker_template(g):
    multi=components(g,False,True)
    def imbalance(ps):
        ct=Counter(g[r][c] for r,c in ps)
        return max(ct.values())-min(ct.values())
    template=max(multi,key=imbalance)
    ct=Counter(g[r][c] for r,c in template)
    if len(ct)!=2:
        raise ValueError('two-color reference')
    marker=min(ct,key=lambda c:(ct[c],c))
    t,l,b,r=bbox(template)
    mpart=template&cells(g,marker)
    mt,ml,_,_=bbox(mpart)
    out=g
    for ps in components(g):
        if colorof(g,ps)!=marker:
            continue
        pt,pl,pb,pr=bbox(ps)
        factor=pr-pl+1
        neighbor=rect(ps,1,True)
        cols={g[i][j] for i,j in neighbor if 0<=i<len(g) and 0<=j<len(g[0])}-{modal(g)}
        if len(cols)!=1:
            raise ValueError('one local paint color')
        col=cols.pop()
        dt,dl=pt-(mt-t)*factor,pl-(ml-l)*factor
        support={(dt+(i-t)*factor+di,dl+(j-l)*factor+dj) for i,j in template for di in range(factor) for dj in range(factor)}
        out=paint(out,support,col)
    return colored(out,{(i,j):v for i,row in enumerate(g) for j,v in enumerate(row) if v!=modal(g)})


def local_template_with_axes(g):
    comps=components(g,False)
    template=min(comps,key=len)
    cr,cc=center(template)
    marker=g[cr][cc]
    t,l,b,r=bbox(template)
    axes=[]
    if b-t>2:
        axes.append(0)
    if r-l>2:
        axes.append(1)
    if not axes:
        raise ValueError('extended reference arms required')
    axial=(cr-1,cc) if 0 in axes else (cr,cc+1)
    col=g[axial[0]][axial[1]]
    bg=modal(g)
    out=paint(g,template,bg)
    for panel in comps:
        if panel==template:
            continue
        marks=panel&cells(g,marker)
        for mr,mc in marks:
            support={(i,j) for i,j in panel if (0 in axes and j==mc) or (1 in axes and i==mr)}
            out=paint(out,support,col)
            assignment={(mr+i-cr,mc+j-cc):g[i][j] for i,j in template if (mr+i-cr,mc+j-cc) in panel}
            out=colored(out,assignment)
    return out


def spiral_canvas(g, dst):
    if len(g)!=len(g[0]) or len(palette(g))!=1:
        raise ValueError('blank square required')
    h,w=shape(g)
    # Clockwise inward walk; leave a one-cell channel between successive turns.
    ps={(0,0)}
    p=(0,0); d=(0,1)
    for _ in range(h*w):
        q=(p[0]+d[0],p[1]+d[1])
        ahead=(p[0]+2*d[0],p[1]+2*d[1])
        if not (0<=q[0]<h and 0<=q[1]<w) or q in ps or ahead in ps:
            d=(d[1],-d[0])
            q=(p[0]+d[0],p[1]+d[1])
            ahead=(p[0]+2*d[0],p[1]+2*d[1])
            if not (0<=q[0]<h and 0<=q[1]<w) or q in ps or ahead in ps:
                break
        ps.add(q);p=q
    return paint(g,ps,dst)


def candidates(demos):
    def propose():
        for fn in (row_palette_template,horizontal_period_complete,lattice_template_complete,scaled_marker_template,local_template_with_axes):
            yield fn,()
        for dst in added_colors(demos):
            yield partial_shape_complete,(dst,)
            yield scaled_diagonal_echo,(dst,2)
            yield spiral_canvas,(dst,)
        # Corner template extent is an input-visible separator or component boundary.
        a,b=demos[0]
        sizes={bbox(p)[3]+1 for p in components(a) if bbox(p)[0]==0 and bbox(p)[1]==0}
        sizes|={c for c in range(1,min(8,len(a[0]))) if len({row[c] for row in a})==1}
        sizes|={n for n in range(1,min(shape(a)))
                if len({a[i][n] for i in range(n+1)}|set(a[n][:n+1]))==1}
        for size in sorted(sizes):
            if size>min(shape(a)):
                continue
            for marker in colors(demos):
                yield corner_stamp,(size,marker)
            for dst in added_colors(demos):
                yield corner_shape_recolor,(size,dst)
    return verified(demos,propose())
