"""A-only conditioned singleton controls with task-visible guards."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def background_of(grid:Grid)->int:return Counter(value for row in grid for value in row).most_common(1)[0][0]
def components(grid:Grid,background:int,connectivity:int=8):
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value!=background};result=[];steps=tuple((dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and (connectivity==8 or abs(dr)+abs(dc)==1))
    while pending:
        seed=pending.pop();item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for dr,dc in steps:
                neighbor=(r+dr,c+dc)
                if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
        result.append(item)
    return result
def render_concentric_depth(grid:Grid,seed_role:int,cycle:tuple[int,...],exceptions:tuple[tuple[int,int,int,int],...])->Grid:
    cells=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==seed_role]
    if not cells or not cycle:raise ValueError('ring seed absent')
    top,bottom=min(r for r,_ in cells),max(r for r,_ in cells);left,right=min(c for _,c in cells),max(c for _,c in cells);height,width=bottom-top+1,right-left+1;overrides={(h,w,depth):color for h,w,depth,color in exceptions};output=[list(row) for row in grid]
    for r in range(top,bottom+1):
        for c in range(left,right+1):
            depth=min(r-top,bottom-r,c-left,right-c);output[r][c]=overrides.get((height,width,depth),cycle[depth%len(cycle)])
    return as_grid(output)
def ring_binding(demos,seed_role):
    observations=[]
    for source,expected in demos:
        cells=[(r,c) for r,row in enumerate(source) for c,value in enumerate(row) if value==seed_role]
        if not cells:return None
        top,bottom=min(r for r,_ in cells),max(r for r,_ in cells);left,right=min(c for _,c in cells),max(c for _,c in cells);height,width=bottom-top+1,right-left+1
        for r in range(top,bottom+1):
            for c in range(left,right+1):observations.append((height,width,min(r-top,bottom-r,c-left,right-c),expected[r][c]))
    best=None
    for length in range(1,9):
        by_remainder={}
        for _,_,depth,color in observations:by_remainder.setdefault(depth%length,Counter())[color]+=1
        if len(by_remainder)<length:continue
        cycle=tuple(by_remainder[index].most_common(1)[0][0] for index in range(length));exceptions=tuple(sorted((h,w,depth,color) for h,w,depth,color in observations if cycle[depth%length]!=color));score=length+3*len(set(exceptions))
        candidate=(score,length,cycle,tuple(sorted(set(exceptions))))
        if best is None or candidate<best:best=candidate
    return None if best is None else (best[2],best[3])
def encode_bar_measurements(grid:Grid,background:int,overrides:tuple[tuple[int,int,int],...])->Grid:
    marker_columns=[c for c,value in enumerate(grid[0]) if value!=background]
    if len(marker_columns)!=1:raise ValueError('reference column absent')
    marker_column=marker_columns[0];lookup={(length,color):code for length,color,code in overrides};output=[list(row) for row in grid]
    for r in range(1,len(grid)):
        cells=[(c,value) for c,value in enumerate(grid[r]) if value!=background]
        if not cells:continue
        colors={value for _,value in cells}
        if len(colors)!=1:raise ValueError('bar not monochrome')
        length=len(cells);color=next(iter(colors));output[r][marker_column]=lookup.get((length,color),((length-1)**2)%10)
    return as_grid(output)
def bar_binding(demos,background):
    overrides={}
    for source,expected in demos:
        marker_columns=[c for c,value in enumerate(source[0]) if value!=background]
        if len(marker_columns)!=1:return None
        marker=marker_columns[0]
        for r in range(1,len(source)):
            cells=[(c,value) for c,value in enumerate(source[r]) if value!=background]
            if not cells:continue
            colors={value for _,value in cells}
            if len(colors)!=1:return None
            length=len(cells);color=next(iter(colors));code=expected[r][marker];base=((length-1)**2)%10
            if code!=base:
                key=(length,color)
                if key in overrides and overrides[key]!=code:return None
                overrides[key]=code
    return tuple(sorted((length,color,code) for (length,color),code in overrides.items()))
def propagate_anchor_templates(grid:Grid,cleanup_signatures:tuple[tuple[int,int,int,int],...])->Grid:
    height,width=len(grid),len(grid[0]);background=background_of(grid);items=components(grid,background,8);templates=[]
    for item in items:
        counts=Counter(grid[r][c] for r,c in item);main,amount=counts.most_common(1)[0]
        if amount>=3 and len(counts)>=2:templates.append((item,main,counts))
    if not templates:raise ValueError('template absent')
    output=[list(row) for row in grid];cleanup=set(cleanup_signatures);copied=0
    for item,main,_ in templates:
        anchors=[(r,c,grid[r][c]) for r,c in item if grid[r][c]!=main];threshold=min(2,len(anchors));translations=set()
        for anchor_row,anchor_column,color in anchors:
            for r,row in enumerate(grid):
                for c,value in enumerate(row):
                    if value==color and (r,c)!=(anchor_row,anchor_column):translations.add((r-anchor_row,c-anchor_column))
        for dr,dc in translations:
            shifted={(r+dr,c+dc) for r,c in item}
            if shifted&item:continue
            matches=0;invalid=False
            for r,c,color in anchors:
                target=(r+dr,c+dc)
                if not (0<=target[0]<height and 0<=target[1]<width):invalid=True;break
                if grid[target[0]][target[1]]==color:matches+=1
                elif grid[target[0]][target[1]]!=background:invalid=True;break
            if invalid or matches<threshold:continue
            copied+=1
            for r,c in item:
                target=(r+dr,c+dc)
                if 0<=target[0]<height and 0<=target[1]<width:output[target[0]][target[1]]=grid[r][c]
    for item,main,counts in templates:
        for role,count in counts.items():
            if (background,main,role,count) in cleanup:
                for r,c in item:
                    if grid[r][c]==role:output[r][c]=background
    if not copied:raise ValueError('confirmed translation absent')
    return as_grid(output)
def cleanup_binding(demos):
    signatures=set()
    for source,expected in demos:
        background=background_of(source)
        for item in components(source,background,8):
            counts=Counter(source[r][c] for r,c in item);main,amount=counts.most_common(1)[0]
            if amount<3 or len(counts)<2:continue
            for role,count in counts.items():
                if any(source[r][c]==role and expected[r][c]==background for r,c in item):signatures.add((background,main,role,count))
    return tuple(sorted(signatures))
def transform_diamond_waves(grid:Grid,shrink:int,expand:int,erased:int)->Grid:
    height,width=len(grid),len(grid[0]);output=[list(row) for row in grid]
    for color,delta in ((shrink,-1),(expand,1)):
        pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color};items=[]
        while pending:
            seed=pending.pop();item={seed};stack=[seed]
            while stack:
                r,c=stack.pop()
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        neighbor=(r+dr,c+dc)
                        if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
            items.append(item)
        for item in items:
            for r,c in item:output[r][c]=erased
            if color==expand and len(item)==3 and max(r for r,_ in item)==height-1:
                row=min(r for r,_ in item);column=next(c for candidate,c in item if candidate==row)
                for target in ((row-1,column+1),(row,column),(row,column+2),(row+1,column-1),(row+1,column+3)):
                    if 0<=target[0]<height and 0<=target[1]<width:output[target[0]][target[1]]=color
                continue
            if len(item)==1:center_row,center_column=next(iter(item));radius=0
            else:
                best=None
                for candidate_row in range(-height,2*height):
                    for candidate_column in range(-width,2*width):
                        distances=[abs(r-candidate_row)+abs(c-candidate_column) for r,c in item];radius,support=Counter(distances).most_common(1)[0];key=(support,-radius,-abs(candidate_row-(height-1)/2)-abs(candidate_column-(width-1)/2))
                        if best is None or key>best[0]:best=(key,candidate_row,candidate_column,radius)
                _,center_row,center_column,radius=best
            radius+=delta
            if radius<0:continue
            for r in range(height):
                remainder=radius-abs(r-center_row)
                if remainder<0:continue
                for c in {center_column-remainder,center_column+remainder}:
                    if 0<=c<width:output[r][c]=color
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='render_concentric_depth':return render_concentric_depth(grid,v['seed_role'],v['cycle'],v['exceptions'])
        if self.graph=='encode_bar_measurements':return encode_bar_measurements(grid,v['background'],v['overrides'])
        if self.graph=='propagate_anchor_templates':return propagate_anchor_templates(grid,v['cleanup_signatures'])
        if self.graph=='transform_diamond_waves':return transform_diamond_waves(grid,v['shrink'],v['expand'],v['erased'])
        raise ValueError('unknown graph')
def candidates(demos):
    if not all(len(source)==len(expected) and len(source[0])==len(expected[0]) for source,expected in demos):return ()
    common=None;source_union=set();output_union=set();result=set();modal_roles=None
    for source,expected in demos:
        palette={value for row in source for value in row};mode={background_of(source)};modal_roles=mode if modal_roles is None else modal_roles&mode;common=palette if common is None else common&palette;source_union.update(palette);output_union.update(value for row in expected for value in row)
    common_colors=tuple(sorted(common or ()))
    if all(len({value for row in source for value in row})==2 for source,_ in demos):
        for seed in common_colors:
            binding=ring_binding(demos,seed)
            if binding is not None:result.add(Candidate('render_concentric_depth',(('cycle',binding[0]),('exceptions',binding[1]),('seed_role',seed))))
    for background in common_colors:
        overrides=bar_binding(demos,background)
        if overrides is not None:result.add(Candidate('encode_bar_measurements',(('background',background),('overrides',overrides))))
    cleanup=cleanup_binding(demos)
    if cleanup:result.add(Candidate('propagate_anchor_templates',(('cleanup_signatures',cleanup),)))
    decreased=[color for color in source_union if all(sum(value==color for row in expected for value in row)<=sum(value==color for row in source for value in row) for source,expected in demos) and any(sum(value==color for row in expected for value in row)<sum(value==color for row in source for value in row) for source,expected in demos)];increased=[color for color in output_union if all(sum(value==color for row in expected for value in row)>=sum(value==color for row in source for value in row) for source,expected in demos) and any(sum(value==color for row in expected for value in row)>sum(value==color for row in source for value in row) for source,expected in demos)]
    if len(decreased)==1 and len(increased)==1:
        for erased in sorted(modal_roles or ()):
            if len({decreased[0],increased[0],erased})==3:result.add(Candidate('transform_diamond_waves',(('erased',erased),('expand',increased[0]),('shrink',decreased[0]))))
    return tuple(sorted(result,key=lambda item:(item.graph,repr(item.bindings))))
def solve(demos,queries,*,budget=2_000_000,allowed_graphs=None):
    allowed=set(allowed_graphs) if allowed_graphs is not None else None;pool=tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed);work=sum(len(a)*len(a[0])+len(b)*len(b[0]) for a,b in demos)*max(1,len(pool))
    if work>budget:return {'status':'abstain_budget','outputs':None,'work':work}
    exact=[]
    for candidate in pool:
        try:
            if all(candidate.execute(source)==expected for source,expected in demos):exact.append(candidate)
        except (IndexError,KeyError,RuntimeError,StopIteration,TypeError,ValueError):continue
    regions={}
    for candidate in exact:
        try:outputs=tuple(candidate.execute(query) for query in queries)
        except (IndexError,KeyError,RuntimeError,StopIteration,TypeError,ValueError):continue
        regions.setdefault(outputs,[]).append(candidate)
    if len(regions)!=1:return {'status':'abstain_no_consensus','outputs':None,'work':work,'regions':len(regions)}
    outputs,members=next(iter(regions.items()));return {'status':'verified','outputs':outputs,'work':work,'regions':1,'graphs':sorted({member.graph for member in members})}
