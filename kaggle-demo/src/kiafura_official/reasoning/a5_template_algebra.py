"""A5 templates are extracted from the current input, never stored in memory."""
from .a5_core import *


def simple_panel_enlarge(g,separator,n):
    objs=components(g,False,True,separator)
    count=lambda o:len({g[r][c] for r,c in o})
    minimum=min(map(count,objs));chosen=[o for o in objs if count(o)==minimum]
    if len(chosen)!=1:raise ValueError('unique simpler panel')
    tile=scale(crop(g,chosen[0]),n)
    out=paste(g,tile,cells(tile))
    return fill(out,separator,cells(g,separator))


def exemplar_copy(g,background):
    objs=components(g,False,False,background)
    count=lambda o:len({g[r][c] for r,c in o})
    richest=max(map(count,objs));chosen=[o for o in objs if count(o)==richest]
    if len(chosen)!=1 or richest<2:raise ValueError('one exemplar')
    template=chosen[0];a,b,_,_=bounds(template);out=g
    for o in objs:
        if o==template:continue
        r,c,_,_=bounds(o);out=paste(out,g,template,(r-a,c-b))
    return out


def framed_shape_recolor(g,frame,background):
    objs=components(g,True,False,background)
    frames=[o for o in objs if o==rectangle(o,edge=True) and all(g[r][c]==frame for r,c in o) and bounds(o)[2]-bounds(o)[0]>1 and bounds(o)[3]-bounds(o)[1]>1]
    if len(frames)!=1:raise ValueError('one closed frame')
    inside=rectangle(frames[0],-1);motif=inside-cells(g,background)
    cs={g[r][c] for r,c in motif}
    if len(cs)!=1:raise ValueError('monochrome key')
    color=cs.pop();key=norm(motif);out=g
    for o in objs:
        if norm(o)==key:out=fill(out,color,o)
    return out


def colored_container_fill(g,frame,background,delta):
    out=g
    for o in components(g,True,False,background):
        if not all(g[r][c]==frame for r,c in o):continue
        interior=rectangle(o)-o
        count=Counter(g[r][c] for r,c in interior)
        if not count:raise ValueError('empty container')
        color=min(count,key=lambda c:(count[c],c))
        out=fill(out,color,move(rectangle(o),delta),under=background)
    return out


def keyed_socket_insert(g,frame,background):
    objs=components(g,False,False,background);loose=[o for o in objs if not o&cells(g,frame)];sockets=[o for o in objs if o&cells(g,frame)]
    if len(loose)!=1 or not sockets:raise ValueError('socket and loose object')
    target=crop(g,set().union(*sockets));q=crop(g,loose[0]);left={c for row in q for c in row[:len(row)//2]}-{background}
    if len(left)!=1:raise ValueError('left role')
    if next(iter(left))!=target[1][0]:q=transform(q,4)
    return paste(target,q,cells(q)-cells(q,background),(1,1))


def keyed_scale_insert(g,frame,background):
    target=crop(g,cells(g,frame));objs=components(g,False,True,background)
    o=max(objs,key=lambda o:bounds(o)[2]);q=crop(g,o)
    anchors=cells(target)-cells(target,frame)-cells(target,background)
    a,b,x,y=bounds(anchors);n=(y-b+1)//len(q[0])
    if n<1:raise ValueError('positive anchor scale')
    q=scale(q,n)
    return paste(target,q,cells(q)-cells(q,background),(a,b))


def keyed_completion(g,background,strategy,erase):
    objs=components(g,False,False,background)
    multicolor=[o for o in objs if len({g[r][c] for r,c in o})>1]
    if not multicolor:raise ValueError('no source motif')
    if strategy=='rare':
        body=max([c for c in palette(g) if c!=background],key=lambda c:len(cells(g,c)))
        keys=set(palette(g))-{background,body};templates=multicolor
    else:
        keys=set.intersection(*({g[r][c] for r,c in o} for o in multicolor))
        richest=max(len({g[r][c] for r,c in o}) for o in multicolor)
        templates=[o for o in multicolor if len({g[r][c] for r,c in o})==richest]
    if not keys:raise ValueError('no key colors')
    paints={};hits=0
    for template in templates:
        tile=crop(g,template)
        for k in range(8):
            q=transform(tile,k);mask={(r,c) for r,c in cells(q) if q[r][c] in keys}
            if not mask:continue
            a,b=min(mask);first=q[a][b]
            for ar,ac in sorted(cells(g,first)):
                dr,dc=ar-a,ac-b
                if not all(0<=r+dr<len(g) and 0<=c+dc<len(g[0]) and g[r+dr][c+dc]==q[r][c] for r,c in mask):continue
                hits+=1
                for r,c in cells(q)-cells(q,background):
                    p=(r+dr,c+dc);color=q[r][c]
                    if p in paints and paints[p]!=color:raise ValueError('inconsistent completions')
                    paints[p]=color
    if not hits:raise ValueError('no occurrences')
    out=g
    for c in sorted(set(paints.values())):out=fill(out,c,{p for p,v in paints.items() if v==c})
    if erase:out=fill(out,background,set().union(*templates))
    return out


def reflected_patch(g,occluder):
    ps=cells(g,occluder)
    if not ps:raise ValueError('occluder')
    options={crop(transform(g,k),ps) for k in (4,6)}
    options={q for q in options if occluder not in palette(q)}
    if len(options)!=1:raise ValueError('reflection ambiguous')
    return options.pop()


def candidates(demos):
    if not demos:return ()
    result=[];inc=colors(demos)
    if same_shape(demos):
        for background in inc:
            offer(result,demos,exemplar_copy,background)
            for strategy,erase in product(('rare','shared'),(False,True)):
                offer(result,demos,keyed_completion,background,strategy,erase)
            for frame in inc:
                if frame==background:continue
                offer(result,demos,framed_shape_recolor,frame,background)
                for delta in [(-1,0),(1,0),(0,-1),(0,1)]:offer(result,demos,colored_container_fill,frame,background,delta)
        for separator in inc:
            for n in (2,3,4):offer(result,demos,simple_panel_enlarge,separator,n)
    else:
        for frame in inc:
            offer(result,demos,reflected_patch,frame)
            for background in inc:
                if background==frame:continue
                offer(result,demos,keyed_socket_insert,frame,background)
                offer(result,demos,keyed_scale_insert,frame,background)
    return tuple(result)
