"""A5 feature classification, counting, object packing and relational layouts."""
from .a5_core import *


def symbol_class(tile,ink):
    n=len(cells(tile,ink));h,w=shape(tile)
    if n==1:return 'singleton'
    if n==h*w-1:return 'one_hole'
    if all(c==ink for c in tile[0]):return 'full_top'
    if all(c==ink for c in tile[-1]):return 'full_bottom'
    return 'remaining'


def symbol_panels(g,ink,mapping):
    h,w=shape(g);table=dict(mapping)
    if w%h:raise ValueError('square symbols')
    return hcat([canvas(table[symbol_class(tuple(row[c:c+h] for row in g),ink)],(h,h)) for c in range(0,w,h)])


def marker_column_labels(g,marker,mapping):
    table=dict(mapping);out=[]
    for row in g:
        positions=[j for j,c in enumerate(row) if c==marker]
        if len(positions)!=1:raise ValueError('one row marker')
        out.append((table[positions[0]],)*len(row))
    return tuple(out)


def histogram_columns(g,background,offset):
    h,w=shape(g);out=canvas(background,(h,w))
    for color in palette(g):
        if color!=background:
            count=len(cells(g,color)); col=color+offset
            if not 0<=col<w:raise ValueError('color column')
            out=fill(out,color,{(r,col) for r in range(max(0,h-count),h)})
    return out


def compressed_color_rank(g):
    h,w=shape(g)
    rows=[r for r in range(h) if len(set(g[r]))>1]
    cols=[c for c in range(w) if len({g[r][c] for r in range(h)})>1]
    q=tuple(tuple(g[r][c] for c in cols) for r in rows)
    count=Counter(c for row in q for c in row)
    order=sorted(count,key=lambda c:(-count[c],c))
    if len(order)!=4:raise ValueError('ranked palette')
    return tuple((c,) for c in order[1:])


def missing_height_snake(g,frame,ink,background,n):
    a,b,x,y=bounds(cells(g,frame));i,j,u,v=bounds(cells(g,ink))
    missing=x-a-(u-i+1)
    if not 0<=missing<=n*n:raise ValueError('height difference')
    out=canvas(background,(n,n))
    ps={(k//n,k%n if (k//n)%2==0 else n-1-k%n) for k in range(missing)}
    return fill(out,ink,ps)


def component_lattice(g,background,minimum):
    objs=[o for o in components(g,True,True,background) if len(o)>minimum]
    points=[(center(o),g[next(iter(o))[0]][next(iter(o))[1]]) for o in objs]
    rows=sorted({p[0] for p,c in points});cols=sorted({p[1] for p,c in points})
    out=canvas(background,(len(rows),len(cols)))
    for (r,c),color in points:out=fill(out,color,{(rows.index(r),cols.index(c))})
    return out


def pack_quadrants(g,background,gap):
    objs=sorted(components(g,True,True,background),key=lambda o:(center(o)[0],center(o)[1]))
    if len(objs)!=4:raise ValueError('four objects')
    objs=sorted(objs[:2],key=lambda o:center(o)[1])+sorted(objs[2:],key=lambda o:center(o)[1])
    heights=[bounds(o)[2]-bounds(o)[0]+1 for o in objs];widths=[bounds(o)[3]-bounds(o)[1]+1 for o in objs]
    th=max(heights[:2]);lh=max(widths[::2]);out=canvas(background,(th+gap+max(heights[2:]),lh+gap+max(widths[1::2])))
    for k,o in enumerate(objs):
        a,b,_,_=bounds(o);out=paste(out,g,o,((th+gap)*(k//2)-a,(lh+gap)*(k%2)-b))
    return out


def bridge_pair(g,background,arm):
    ps=cells(g)-cells(g,background);a,b,x,y=bounds(ps);transpose=x-a<y-b
    if transpose:g=transform(g,5)
    objs=sorted(components(g,True,False,background),key=lambda o:center(o)[0])
    if len(objs)!=2 or any(len(o)!=1 for o in objs):raise ValueError('two seeds')
    p,q=[next(iter(o)) for o in objs]
    if p[1]!=q[1]:raise ValueError('aligned points')
    c1,c2=g[p[0]][p[1]],g[q[0]][q[1]];m=(p[0]+q[0])//2;col=p[1]
    out=fill(g,c2,line(p,q));out=fill(out,c1,line(p,(m,col)))
    out=fill(out,c1,{(m,col-arm),(m,col+arm)}|{(m-1,c) for c in range(col-arm,col+arm+1)})
    out=fill(out,c2,{(m+1,col-arm),(m+1,col+arm)}|{(m+2,c) for c in range(col-arm,col+arm+1)})
    out=fill(out,background,{(m,col),(m+1,col)})
    return transform(out,5) if transpose else out


def candidates(demos):
    if not demos:return ()
    result=[];inc=colors(demos);outc=colors(demos,1)
    h,w=shape(demos[0][0]);ho,wo=shape(demos[0][1])
    if same_shape(demos):
        for ink in inc:
            mapping={};valid=True
            for i,o in demos:
                n=len(i)
                if len(i[0])%n:valid=False;break
                for c in range(0,len(i[0]),n):
                    tile=tuple(row[c:c+n] for row in i);dest=tuple(row[c:c+n] for row in o)
                    if len(palette(dest))!=1:valid=False;break
                    key=symbol_class(tile,ink);v=dest[0][0]
                    if key in mapping and mapping[key]!=v:valid=False;break
                    mapping[key]=v
                if not valid:break
            if valid:offer(result,demos,symbol_panels,ink,tuple(sorted(mapping.items())))
            mapping={};valid=True
            for i,o in demos:
                for row,dest in zip(i,o):
                    ps=[j for j,c in enumerate(row) if c==ink]
                    if len(ps)!=1 or len(set(dest))!=1:valid=False;break
                    k=ps[0];v=dest[0]
                    if k in mapping and mapping[k]!=v:valid=False;break
                    mapping[k]=v
                if not valid:break
            if valid:offer(result,demos,marker_column_labels,ink,tuple(sorted(mapping.items())))
        for background in inc:
            for offset in (-1,0):offer(result,demos,histogram_columns,background,offset)
            for arm in (1,2,3):offer(result,demos,bridge_pair,background,arm)
    if ho<h or wo<w:
        offer(result,demos,compressed_color_rank)
        for background in inc:
            for minimum in (1,2):offer(result,demos,component_lattice,background,minimum)
            for gap in (0,1,2):offer(result,demos,pack_quadrants,background,gap)
        if ho==wo and ho<=5:
            for frame,ink in permutations(inc,2):
                for background in outc:offer(result,demos,missing_height_snake,frame,ink,background,ho)
    return tuple(result)
