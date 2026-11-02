"""Typed propagation and recoloring operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def tile_mark_diagonals(grid:Grid,background:int,marker:int,scale:int)->Grid:
    height,width=len(grid),len(grid[0]);output=[[grid[r%height][c%width] for c in range(scale*width)] for r in range(scale*height)];seeds=[(r,c) for r,row in enumerate(output) for c,value in enumerate(row) if value!=background]
    for r,c in seeds:
        for dr,dc in product((-1,1),repeat=2):
            rr,cc=r+dr,c+dc
            if 0<=rr<len(output) and 0<=cc<len(output[0]) and output[rr][cc]==background:output[rr][cc]=marker
    return as_grid(output)
def recolor_diagonal_orientations(grid:Grid,source:int,dominant:int,minority:int)->Grid:
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==source};items=[]
    while pending:
        seed=min(pending);pending.remove(seed);item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for dr,dc in product((-1,1),repeat=2):
                point=r+dr,c+dc
                if point in pending:pending.remove(point);item.add(point);stack.append(point)
        ordered=sorted(item);items.append((1 if ordered[-1][1]>ordered[0][1] else -1,item))
    if not items:raise ValueError('diagonal components absent')
    support=Counter(orientation for orientation,item in items for _ in item);winner=min(support,key=lambda value:(-support[value],value));output=[list(row) for row in grid]
    for orientation,item in items:
        for r,c in item:output[r][c]=dominant if orientation==winner else minority
    return as_grid(output)
def continue_progression(grid:Grid,seed:int,ink:int)->Grid:
    points=sorted((r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==seed)
    if len(points)<2:raise ValueError('at least two progression points required')
    dr,dc=points[1][0]-points[0][0],points[1][1]-points[0][1]
    if (dr,dc)==(0,0):raise ValueError('zero progression')
    if any((b[0]-a[0],b[1]-a[1])!=(dr,dc) for a,b in zip(points,points[1:])):raise ValueError('inconsistent progression')
    output=[list(row) for row in grid];r,c=points[-1][0]+dr,points[-1][1]+dc
    while 0<=r<len(grid) and 0<=c<len(grid[0]):output[r][c]=ink;r+=dr;c+=dc
    return as_grid(output)
def write_mode_below_separator(grid:Grid,background:int)->Grid:
    separator=next(r for r,row in enumerate(grid) if len(set(row))==1 and row[0]!=background);mode=Counter(value for row in grid[:separator] for value in row).most_common(1)[0][0];output=[list(row) for row in grid];output[-1][len(grid[0])//2]=mode;return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='tile_mark_diagonals':return tile_mark_diagonals(grid,v['background'],v['marker'],v['scale'])
        if self.graph=='recolor_diagonal_orientations':return recolor_diagonal_orientations(grid,v['source'],v['dominant'],v['minority'])
        if self.graph=='continue_progression':return continue_progression(grid,v['seed'],v['ink'])
        if self.graph=='write_mode_below_separator':return write_mode_below_separator(grid,v['background'])
        raise ValueError('unknown graph')
def candidates(demos):
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));backgrounds=tuple(sorted({Counter(value for row in source for value in row).most_common(1)[0][0] for source,_ in demos}));added=tuple(sorted({value for _,expected in demos for row in expected for value in row}-{value for source,_ in demos for row in source for value in row}));same=all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos);result=set()
    if same:
        for source,dominant,minority in product(colors,colors,colors):
            if len({source,dominant,minority})==3:result.add(Candidate('recolor_diagonal_orientations',tuple(sorted({'dominant':dominant,'minority':minority,'source':source}.items()))))
        for seed,ink in product(colors,added or colors):
            if seed!=ink:result.add(Candidate('continue_progression',(('ink',ink),('seed',seed))))
        for background in backgrounds:result.add(Candidate('write_mode_below_separator',(('background',background),)))
    else:
        scales={(len(expected)//len(source),len(expected[0])//len(source[0])) for source,expected in demos if len(expected)%len(source)==0 and len(expected[0])%len(source[0])==0}
        if len(scales)==1:
            sr,sc=next(iter(scales))
            if sr==sc:
                for background,marker in product(backgrounds,added or colors):
                    if background!=marker:result.add(Candidate('tile_mark_diagonals',tuple(sorted({'background':background,'marker':marker,'scale':sr}.items()))))
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
