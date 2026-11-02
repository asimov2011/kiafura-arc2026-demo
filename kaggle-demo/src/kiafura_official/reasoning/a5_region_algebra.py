"""A5 components: selection, topology, morphology and feature-bound recolor."""
from .a5_core import *


def region_feature(g,ps,mode):
    a,b,x,y=bounds(ps); h,w=shape(g)
    if mode=='size': return len(ps)
    if mode=='singleton': return int(len(ps)==1)
    if mode=='small': return int(len(ps)<=2)
    if mode=='border': return int(a==0 or b==0 or x==h-1 or y==w-1)
    if mode=='square_parity':
        return (x-a+1)%2 if x-a==y-b and len(ps)==(x-a+1)*(y-b+1) else -1
    raise ValueError('feature')


def component_recolor(g,background,diagonal,target,feature,mapping):
    table=dict(mapping);out=g
    for ps in components(g,True,diagonal,background,True):
        color=g[next(iter(ps))[0]][next(iter(ps))[1]]
        if (target=='foreground' and color==background) or (target!='foreground' and color!=target):continue
        key=region_feature(g,ps,feature)
        if key in table:out=fill(out,table[key],ps)
    return out


def select_crop(g,mono,diagonal,selector,color,quarter):
    objs=components(g,mono,diagonal)
    def score(ps):
        if selector=='size':return len(ps)
        if selector=='colors':return len({g[r][c] for r,c in ps})
        if selector=='colorcount':return len(ps&cells(g,color))
        if selector=='symmetric':
            q=crop(g,ps)
            return int(q==transform(q,4))
        raise ValueError('selector')
    best=max(map(score,objs))
    selected=[o for o in objs if score(o)==best]
    if len(selected)!=1:raise ValueError('ambiguous selection')
    q=crop(g,selected[0])
    return tuple(row[:len(q[0])//2] for row in q[:len(q)//2]) if quarter else q


def extreme_regions(g,background,large,small):
    objs=components(g,True,False,background,True)
    sizes=[len(o) for o in objs]
    out=g
    for o in objs:
        if len(o)==max(sizes):out=fill(out,large,o)
        if len(o)==min(sizes) and all(g[r][c]==background for r,c in o):out=fill(out,small,o)
    return out


def ranked_squares(g,background,single,medium,large):
    objs=[o for o in components(g,True,False,background,True) if all(g[r][c]==background for r,c in o) and region_feature(g,o,'square_parity')!=-1]
    biggest=max(map(len,objs));out=g
    for o in objs:out=fill(out,single if len(o)==1 else large if len(o)==biggest else medium,o)
    return out


def ordered_regions(g,background,mark,period,phase):
    objs=[o for o in components(g,True,False,background,True) if all(g[r][c]==background for r,c in o)]
    objs.sort(key=lambda o:(center(o)[1],center(o)[0]));out=g
    for i,o in enumerate(objs):
        if i%period==phase:out=fill(out,mark,o)
    return out


def touching_flood(g,background,seed):
    marks=cells(g,seed);out=g
    adjacent={p for r,c in marks for p in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]}
    for o in components(g,True,False,background,True):
        if o&adjacent and all(g[r][c]==background for r,c in o):out=fill(out,seed,o)
    return out


def bounding_frame(g,background,mark):
    return fill(g,mark,rectangle(cells(g)-cells(g,background),edge=True),under=background)


def empty_bbox_frame(g,background):
    objs=components(g,True,False,background,True)
    singles=[o for o in objs if len(o)==1]
    colorset={g[r][c] for o in singles for r,c in o}
    if len(colorset)!=1:raise ValueError('marker color')
    return fill(canvas(background,shape(g)),colorset.pop(),rectangle(cells(g,background),edge=True))


def clip_color_boxes(g,frame,background):
    ps=rectangle(cells(g,frame));out=fill(canvas(background,shape(g)),frame,ps)
    for c in palette(g):
        if c not in (frame,background):out=fill(out,c,rectangle(cells(g,c))&ps)
    return out


def isolated_erosion(g,foreground,background,threshold):
    if foreground=='least':foreground=min(palette(g),key=lambda c:len(cells(g,c)))
    ps=cells(g,foreground)
    gone={p for p in ps if sum((p[0]+dr,p[1]+dc) not in ps for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])>threshold}
    return fill(g,background,gone)


def paired_recolor(g,background,trigger,partner,mark):
    out=g
    for o in components(g,False,False,background):
        if len(o)==2 and trigger in {g[r][c] for r,c in o}:
            out=fill(out,background,o)
            out=fill(out,mark,o&cells(g,partner))
    return out


def other_than_largest(g,background):
    objs=components(g,True,False,background)
    ps=max(objs,key=len);colorset=set(palette(g))-{background,g[next(iter(ps))[0]][next(iter(ps))[1]]}
    if len(colorset)!=1:raise ValueError('other color')
    return ((colorset.pop(),),)


def candidates(demos):
    if not demos:return ()
    result=[]; inc=colors(demos);outc=colors(demos,1)
    h,w=shape(demos[0][0]);ho,wo=shape(demos[0][1])
    if ho<=h and wo<=w and (ho<h or wo<w):
        for mono,diagonal,quarter in product((False,True),repeat=3):
            for selector in ('size','colors','symmetric'):
                offer(result,demos,select_crop,mono,diagonal,selector,0,quarter)
            for c in inc:offer(result,demos,select_crop,mono,diagonal,'colorcount',c,quarter)
        if (ho,wo)==(1,1):
            for background in inc:offer(result,demos,other_than_largest,background)
    if not same_shape(demos):return tuple(result)
    for background in inc:
        for diagonal in (False,True):
            for target in [c for c in inc]+['foreground']:
                for feature in ('singleton','small','border','square_parity'):
                    mapping={};valid=True;changed=False
                    for i,o in demos:
                        for ps in components(i,True,diagonal,background,True):
                            color=i[next(iter(ps))[0]][next(iter(ps))[1]]
                            if (target=='foreground' and color==background) or (target!='foreground' and color!=target):continue
                            key=region_feature(i,ps,feature);vals={o[r][c] for r,c in ps}
                            if len(vals)!=1:valid=False;break
                            v=vals.pop()
                            # Unchanged foreground colors must remain input-relative.
                            if v==color:continue
                            if key in mapping and mapping[key]!=v:valid=False;break
                            mapping[key]=v;changed=True
                        if not valid:break
                    if valid and changed:offer(result,demos,component_recolor,background,diagonal,target,feature,tuple(sorted(mapping.items())))
        offer(result,demos,empty_bbox_frame,background)
        for c in outc:
            offer(result,demos,bounding_frame,background,c)
            if c in inc:offer(result,demos,touching_flood,background,c)
            for period in range(2,5):
                for phase in range(period):offer(result,demos,ordered_regions,background,c,period,phase)
        for large,small in permutations(outc,2):offer(result,demos,extreme_regions,background,large,small)
        for single,medium,large in permutations(sorted({c for i,o in demos for c in palette(o)}),3):offer(result,demos,ranked_squares,background,single,medium,large)
        offer(result,demos,isolated_erosion,'least',background,2)
        for foreground in inc:
            offer(result,demos,clip_color_boxes,foreground,background)
            offer(result,demos,isolated_erosion,foreground,background,2)
        # Pair roles bound to colors that disappear/change in demonstrations.
        if len(inc)<=4:
            for trigger,partner in permutations([c for c in inc if c!=background],2):
                for mark in outc:offer(result,demos,paired_recolor,background,trigger,partner,mark)
    return tuple(result)
