"""Typed geometric pattern operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def complete_alternating_triangle(grid:Grid,primary:int,secondary:int)->Grid:
    points=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==primary]
    if not points:raise ValueError('trunk absent')
    axis=Counter(c for _,c in points).most_common(1)[0][0];apex=max(r for r,c in points if c==axis);output=[list(row) for row in grid]
    for r in range(apex):
        radius=apex-r
        for c in range(max(0,axis-radius),min(len(grid[0]),axis+radius+1)):output[r][c]=primary if (c-axis)%2==0 else secondary
    return as_grid(output)
def propagate_chebyshev_rings(grid:Grid,background:int)->Grid:
    cr,cc=len(grid)//2,len(grid[0])//2;rings={max(abs(r-cr),abs(c-cc)):value for r,row in enumerate(grid) for c,value in enumerate(row) if value!=background}
    if not rings:raise ValueError('ring evidence absent')
    return as_grid([[rings.get(max(abs(r-cr),abs(c-cc)),background) for c in range(len(grid[0]))] for r in range(len(grid))])
def concentric_squares_from_points(grid:Grid,background:int)->Grid:
    points=sorted((r,c,value) for r,row in enumerate(grid) for c,value in enumerate(row) if value!=background)
    if len(points)!=3:raise ValueError('three points required')
    first,center,_=points;radius=max(abs(first[0]-center[0]),abs(first[1]-center[1]));
    if radius<=0:raise ValueError('positive radius required')
    color=first[2];output=[[background]*len(grid[0]) for _ in grid];output[center[0]][center[1]]=color;limit=max(center[0],len(grid)-1-center[0],center[1],len(grid[0])-1-center[1]);current=radius
    while current<=limit:
        top,bottom,left,right=center[0]-current,center[0]+current,center[1]-current,center[1]+current
        for c in range(max(0,left),min(len(grid[0])-1,right)+1):
            if 0<=top<len(grid):output[top][c]=color
            if 0<=bottom<len(grid):output[bottom][c]=color
        for r in range(max(0,top),min(len(grid)-1,bottom)+1):
            if 0<=left<len(grid[0]):output[r][left]=color
            if 0<=right<len(grid[0]):output[r][right]=color
        current+=radius
    return as_grid(output)
def alternate_row_shifts(grid:Grid,background:int,first_direction:str)->Grid:
    occupied=[r for r,row in enumerate(grid) if any(value!=background for value in row)]
    if len(occupied)<2:raise ValueError('row pattern absent')
    output=[list(row) for row in grid];left=first_direction=='left'
    for r in range(max(occupied)-1,min(occupied)-1,-2):
        output[r]=list(grid[r][1:]+(background,)) if left else list((background,)+grid[r][:-1]);left=not left
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='complete_alternating_triangle':return complete_alternating_triangle(grid,v['primary'],v['secondary'])
        if self.graph=='propagate_chebyshev_rings':return propagate_chebyshev_rings(grid,v['background'])
        if self.graph=='concentric_squares_from_points':return concentric_squares_from_points(grid,v['background'])
        if self.graph=='alternate_row_shifts':return alternate_row_shifts(grid,v['background'],v['first_direction'])
        raise ValueError('unknown graph')
def candidates(demos):
    if not all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos):return ()
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));backgrounds=tuple(sorted({Counter(value for row in source for value in row).most_common(1)[0][0] for source,_ in demos}));result=set()
    for primary,secondary in product(colors,colors):
        if primary!=secondary:result.add(Candidate('complete_alternating_triangle',(('primary',primary),('secondary',secondary))))
    for background in backgrounds:
        result.add(Candidate('propagate_chebyshev_rings',(('background',background),)));result.add(Candidate('concentric_squares_from_points',(('background',background),)))
        for direction in ('left','right'):result.add(Candidate('alternate_row_shifts',(('background',background),('first_direction',direction))))
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
