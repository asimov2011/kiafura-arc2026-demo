"""Typed local-pattern control operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def expand_cross_prototypes(grid:Grid,background:int,radius:int)->Grid:
    height,width=len(grid),len(grid[0]);output=[list(row) for row in grid];centers=[]
    for r in range(1,height-1):
        for c in range(1,width-1):
            center=grid[r][c];arms=(grid[r-1][c],grid[r+1][c],grid[r][c-1],grid[r][c+1])
            if center!=background and arms[0]!=background and len(set(arms))==1:centers.append((r,c,center,arms[0]))
    if not centers:raise ValueError('cross prototype absent')
    for r,c,center,arm in centers:
        for delta in range(-radius,radius+1):
            for rr,cc in ((r+delta,c),(r,c+delta)):
                if 0<=rr<height and 0<=cc<width:output[rr][cc]=arm
            for rr,cc in ((r+delta,c+delta),(r+delta,c-delta)):
                if 0<=rr<height and 0<=cc<width:output[rr][cc]=center
    return as_grid(output)
def recolor_center_with_rays(grid:Grid,center:int,arm:int,replacement:int,inner_gap:int,ray_length:int)->Grid:
    output=[list(row) for row in grid];changed=0
    margin=inner_gap+ray_length-1
    for r in range(margin,len(grid)-margin):
        for c in range(margin,len(grid[0])-margin):
            if grid[r][c]!=center:continue
            rays=tuple(tuple((r+dr*k,c+dc*k) for k in range(inner_gap,inner_gap+ray_length)) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)))
            if all(all(grid[rr][cc]==arm for rr,cc in ray) for ray in rays):output[r][c]=replacement;changed+=1
    if not changed:raise ValueError('ray center absent')
    return as_grid(output)
def complete_cross_coverage(grid:Grid,pattern:int,target:int,replacement:int,link_distance:int)->Grid:
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==pattern};groups=[]
    while pending:
        seed=min(pending);pending.remove(seed);item={seed};stack=[seed]
        while stack:
            r,c=stack.pop();near=[point for point in pending if abs(point[0]-r)+abs(point[1]-c)<=link_distance]
            for point in near:pending.remove(point);item.add(point);stack.append(point)
        groups.append(item)
    if not groups:raise ValueError('cross groups absent')
    descriptions=[]
    for item in groups:
        rows=Counter(r for r,_ in item);columns=Counter(c for _,c in item)
        if len(rows)==1:r=next(iter(rows));c=(min(c for _,c in item)+max(c for _,c in item))//2
        elif len(columns)==1:c=next(iter(columns));r=(min(r for r,_ in item)+max(r for r,_ in item))//2
        else:r=min(rows,key=lambda value:(-rows[value],value));c=min(columns,key=lambda value:(-columns[value],value))
        descriptions.append((r,c,max(max(abs(rr-r),abs(cc-c)) for rr,cc in item)))
    radius=max(value for _,_,value in descriptions);output=[list(row) for row in grid]
    for r,c,_ in descriptions:
        for rr,cc in ({(r+d,c) for d in range(-radius,radius+1)}|{(r,c+d) for d in range(-radius,radius+1)}):
            if 0<=rr<len(grid) and 0<=cc<len(grid[0]) and output[rr][cc]==target:output[rr][cc]=replacement
    return as_grid(output)
def recolor_header_parity(grid:Grid,source:int,replacement:int)->Grid:
    parity=1-sum(c for c,value in enumerate(grid[0]) if value==source)%2;output=[list(row) for row in grid]
    for r,row in enumerate(grid):
        for c,value in enumerate(row):
            if value==source and c%2==parity:output[r][c]=replacement
    return as_grid(output)
def annotate_nearest_full_axis(grid:Grid,background:int,marker:int)->Grid:
    height,width=len(grid),len(grid[0]);full_rows=[(r,grid[r][0]) for r in range(height) if grid[r][0]!=background and len(set(grid[r]))==1];full_columns=[(c,grid[0][c]) for c in range(width) if grid[0][c]!=background and all(grid[r][c]==grid[0][c] for r in range(height))];markers=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==marker];output=[list(row) for row in grid]
    if len(full_rows)>=2:
        first,last=full_rows[0],full_rows[-1]
        for r,c in markers:
            rr,color=(r-1,first[1]) if abs(r-first[0])<abs(r-last[0]) else (r+1,last[1])
            if 0<=rr<height and output[rr][c]==background:output[rr][c]=color
    elif len(full_columns)>=2:
        first,last=full_columns[0],full_columns[-1]
        for r,c in markers:
            cc,color=(c-1,first[1]) if abs(c-first[0])<abs(c-last[0]) else (c+1,last[1])
            if 0<=cc<width and output[r][cc]==background:output[r][cc]=color
    else:raise ValueError('full axes absent')
    return as_grid(output)
def cross_marker_lines(grid:Grid,marker_a:int,marker_b:int,intersection:int)->Grid:
    markers=[(r,c,value) for r,row in enumerate(grid) for c,value in enumerate(row) if value in (marker_a,marker_b)];output=[list(row) for row in grid]
    if len(markers)<2:raise ValueError('markers absent')
    for r,c,color in markers:
        for cc in range(len(grid[0])):output[r][cc]=color
        for rr in range(len(grid)):output[rr][c]=color
    for index,(r,c,_) in enumerate(markers):
        for rr,cc,_ in markers[index+1:]:output[r][cc]=intersection;output[rr][c]=intersection
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='expand_cross_prototypes':return expand_cross_prototypes(grid,v['background'],v['radius'])
        if self.graph=='recolor_center_with_rays':return recolor_center_with_rays(grid,v['center'],v['arm'],v['replacement'],v['inner_gap'],v['ray_length'])
        if self.graph=='complete_cross_coverage':return complete_cross_coverage(grid,v['pattern'],v['target'],v['replacement'],v['link_distance'])
        if self.graph=='recolor_header_parity':return recolor_header_parity(grid,v['source'],v['replacement'])
        if self.graph=='annotate_nearest_full_axis':return annotate_nearest_full_axis(grid,v['background'],v['marker'])
        if self.graph=='cross_marker_lines':return cross_marker_lines(grid,v['marker_a'],v['marker_b'],v['intersection'])
        raise ValueError('unknown graph')
def candidates(demos):
    if not all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos):return ()
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));backgrounds=tuple(sorted({Counter(value for row in source for value in row).most_common(1)[0][0] for source,_ in demos}));added=tuple(sorted({value for _,expected in demos for row in expected for value in row}-{value for source,_ in demos for row in source for value in row}));result=set()
    for background in backgrounds:
        for radius in range(1,4):result.add(Candidate('expand_cross_prototypes',(('background',background),('radius',radius))))
        for marker in colors:
            if marker!=background:result.add(Candidate('annotate_nearest_full_axis',tuple(sorted({'background':background,'marker':marker}.items()))))
    for center,arm,replacement in product(colors,colors,added or colors):
        if len({center,arm,replacement})==3:result.add(Candidate('recolor_center_with_rays',tuple(sorted({'arm':arm,'center':center,'inner_gap':2,'ray_length':2,'replacement':replacement}.items()))))
    for pattern,target,replacement in product(colors,colors,added or colors):
        if len({pattern,target,replacement})==3:result.add(Candidate('complete_cross_coverage',tuple(sorted({'link_distance':4,'pattern':pattern,'replacement':replacement,'target':target}.items()))))
    for source,replacement in product(colors,added or colors):
        if source!=replacement:result.add(Candidate('recolor_header_parity',(('replacement',replacement),('source',source))))
    for marker_a,marker_b,intersection in product(colors,colors,added or colors):
        if len({marker_a,marker_b,intersection})==3:result.add(Candidate('cross_marker_lines',tuple(sorted({'intersection':intersection,'marker_a':min(marker_a,marker_b),'marker_b':max(marker_a,marker_b)}.items()))))
    return tuple(sorted(result,key=lambda item:(item.graph,repr(item.bindings))))
def solve(demos,queries,*,budget=2_000_000,allowed_graphs=None):
    allowed=set(allowed_graphs) if allowed_graphs is not None else None;pool=tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed);work=sum(len(a)*len(a[0])+len(b)*len(b[0]) for a,b in demos)*max(1,len(pool))
    if work>budget:return {'status':'abstain_budget','outputs':None,'work':work}
    exact=[]
    for candidate in pool:
        try:
            if all(candidate.execute(source)==expected for source,expected in demos):exact.append(candidate)
        except (IndexError,KeyError,TypeError,ValueError):continue
    regions={}
    for candidate in exact:
        try:outputs=tuple(candidate.execute(query) for query in queries)
        except (IndexError,KeyError,TypeError,ValueError):continue
        regions.setdefault(outputs,[]).append(candidate)
    if len(regions)!=1:return {'status':'abstain_no_consensus','outputs':None,'work':work,'regions':len(regions)}
    outputs,members=next(iter(regions.items()));return {'status':'verified','outputs':outputs,'work':work,'regions':1,'graphs':sorted({member.graph for member in members})}
