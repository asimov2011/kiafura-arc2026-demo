"""Typed panel composition operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def split_panels(grid:Grid,separator:int):
    height,width=len(grid),len(grid[0]);rows=[r for r,row in enumerate(grid) if all(value==separator for value in row)];columns=[c for c in range(width) if all(grid[r][c]==separator for r in range(height))];rb=[-1,*rows,height];cb=[-1,*columns,width];return tuple((rb[i]+1,rb[i+1],cb[j]+1,cb[j+1]) for i in range(len(rb)-1) for j in range(len(cb)-1))
def replicate_nonempty_panel(grid:Grid,background:int,separator:int)->Grid:
    panels=split_panels(grid,separator)
    if len(panels)<2:raise ValueError('multiple panels required')
    source=max(panels,key=lambda box:sum(grid[r][c]!=background for r in range(box[0],box[1]) for c in range(box[2],box[3])));pattern=tuple(tuple(grid[r][source[2]:source[3]]) for r in range(source[0],source[1]));output=[list(row) for row in grid]
    for top,bottom,left,right in panels:
        if bottom-top==len(pattern) and right-left==len(pattern[0]):
            for offset,row in enumerate(pattern):output[top+offset][left:right]=row
    return as_grid(output)
def stamp_pattern_at_marker(grid:Grid,background:int,marker:int)->Grid:
    markers=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==marker]
    if len(markers)!=1:raise ValueError('single marker required')
    cells=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value not in (background,marker)]
    if not cells:raise ValueError('pattern absent')
    top,bottom=min(r for r,_ in cells),max(r for r,_ in cells);left,right=min(c for _,c in cells),max(c for _,c in cells);pattern=tuple(tuple(grid[r][left:right+1]) for r in range(top,bottom+1));origin_r=markers[0][0]-len(pattern)//2;origin_c=markers[0][1]-len(pattern[0])//2;output=[list(row) for row in grid]
    for r,row in enumerate(pattern):
        for c,value in enumerate(row):
            rr,cc=origin_r+r,origin_c+c
            if value!=background and 0<=rr<len(grid) and 0<=cc<len(grid[0]):output[rr][cc]=value
    return as_grid(output)
def compress_cross_panels(grid:Grid)->Grid:
    height,width=len(grid),len(grid[0]);row=next(r for r,values in enumerate(grid) if len(set(values))==1);separator=grid[row][0];column=next(c for c in range(width) if all(grid[r][c]==separator for r in range(height)))
    def mode(top,bottom,left,right):return Counter(grid[r][c] for r in range(top,bottom) for c in range(left,right)).most_common(1)[0][0]
    return as_grid(((mode(0,row,0,column),separator,mode(0,row,column+1,width)),(separator,separator,separator),(mode(row+1,height,0,column),separator,mode(row+1,height,column+1,width))))
def relocate_separator(grid:Grid,background:int,separator:int,source_role:int,target_role:int,preserve_count:int)->Grid:
    height,width=len(grid),len(grid[0]);column=next(c for c in range(width) if all(grid[r][c]==separator for r in range(height)));destination=2*column+1
    if destination>=width:raise ValueError('destination outside canvas')
    output=[list(row) for row in grid]
    for r in range(height):output[r][column]=background;output[r][destination]=separator
    count=sum(grid[r][c]==target_role for r in range(height) for c in range(column))
    if count!=preserve_count:
        for r in range(height):
            for c in range(column+1,destination):
                if output[r][c]==source_role:output[r][c]=target_role
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='replicate_nonempty_panel':return replicate_nonempty_panel(grid,v['background'],v['separator'])
        if self.graph=='stamp_pattern_at_marker':return stamp_pattern_at_marker(grid,v['background'],v['marker'])
        if self.graph=='compress_cross_panels':return compress_cross_panels(grid)
        if self.graph=='relocate_separator':return relocate_separator(grid,v['background'],v['separator'],v['source_role'],v['target_role'],v['preserve_count'])
        raise ValueError('unknown graph')
def candidates(demos):
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));backgrounds=tuple(sorted({Counter(value for row in source for value in row).most_common(1)[0][0] for source,_ in demos}));same=all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos);result=set()
    if same:
        for background,separator in product(backgrounds,colors):
            if background!=separator:result.add(Candidate('replicate_nonempty_panel',tuple(sorted({'background':background,'separator':separator}.items()))))
        for background,marker in product(backgrounds,colors):
            if background!=marker:result.add(Candidate('stamp_pattern_at_marker',tuple(sorted({'background':background,'marker':marker}.items()))))
        for background,separator,source_role,target_role,preserve_count in product(backgrounds,colors,colors,colors,range(0,4)):
            if len({background,separator,source_role,target_role})==4:result.add(Candidate('relocate_separator',tuple(sorted({'background':background,'preserve_count':preserve_count,'separator':separator,'source_role':source_role,'target_role':target_role}.items()))))
    else:
        result.add(Candidate('compress_cross_panels',()))
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
