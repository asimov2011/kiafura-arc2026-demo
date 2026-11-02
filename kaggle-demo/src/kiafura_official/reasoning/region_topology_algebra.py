"""Typed region/topology operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter,deque
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)

def expand_seed_quadrants(grid:Grid,background:int)->Grid:
    cells=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value!=background]
    if not cells:raise ValueError('seed absent')
    top,left=min(r for r,_ in cells),min(c for _,c in cells)
    if top+1>=len(grid) or left+1>=len(grid[0]):raise ValueError('2x2 seed absent')
    colors=(grid[top][left],grid[top][left+1],grid[top+1][left],grid[top+1][left+1]);output=[list(row) for row in grid]
    for origin_r,origin_c,color in ((top-2,left-2,colors[3]),(top-2,left+2,colors[2]),(top+2,left-2,colors[1]),(top+2,left+2,colors[0])):
        for r in range(origin_r,origin_r+2):
            for c in range(origin_c,origin_c+2):
                if 0<=r<len(grid) and 0<=c<len(grid[0]):output[r][c]=color
    return as_grid(output)

def project_markers_to_separator(grid:Grid,background:int,separator:int)->Grid:
    height,width=len(grid),len(grid[0]);points=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==separator]
    if not points:raise ValueError('separator absent')
    vertical=len({r for r,_ in points})>=len({c for _,c in points});marker_colors={value for row in grid for value in row if value not in (background,separator)};output=[[background if value in marker_colors else value for value in row] for row in grid]
    for r,c in points:
        for color in marker_colors:
            if vertical:
                if c>0 and any(grid[r][x]==color for x in range(c)):output[r][c-1]=color
                if c+1<width and any(grid[r][x]==color for x in range(c+1,width)):output[r][c+1]=color
            else:
                if r>0 and any(grid[x][c]==color for x in range(r)):output[r-1][c]=color
                if r+1<height and any(grid[x][c]==color for x in range(r+1,height)):output[r+1][c]=color
    return as_grid(output)

def partition_regions(grid:Grid,separator:int)->tuple[tuple[tuple[int,int],tuple[tuple[int,int],...]],...]:
    height,width=len(grid),len(grid[0]);rows=[r for r,row in enumerate(grid) if all(value==separator for value in row)];columns=[c for c in range(width) if all(grid[r][c]==separator for r in range(height))]
    row_bounds=[-1,*rows,height];column_bounds=[-1,*columns,width];regions=[]
    for ri in range(len(row_bounds)-1):
        for ci in range(len(column_bounds)-1):regions.append(((ri,ci),tuple((r,c) for r in range(row_bounds[ri]+1,row_bounds[ri+1]) for c in range(column_bounds[ci]+1,column_bounds[ci+1]))))
    return tuple(regions)

def fill_partition_regions(grid:Grid,separator:int,mapping:tuple[tuple[tuple[int,int],int],...])->Grid:
    output=[list(row) for row in grid];regions=dict(partition_regions(grid,separator));
    for coordinate,color in mapping:
        for r,c in regions.get(coordinate,()):output[r][c]=color
    return as_grid(output)

def components(grid:Grid,color:int)->tuple[frozenset[tuple[int,int]],...]:
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color};result=[]
    while pending:
        seed=min(pending);pending.remove(seed);item={seed};queue=deque([seed])
        while queue:
            r,c=queue.popleft()
            for point in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if point in pending:pending.remove(point);item.add(point);queue.append(point)
        result.append(frozenset(item))
    return tuple(result)

def fill_ranked_rectangles(grid:Grid,border:int,rank_colors:tuple[int,...])->Grid:
    boxes=[]
    for item in components(grid,border):
        top,bottom=min(r for r,_ in item),max(r for r,_ in item);left,right=min(c for _,c in item),max(c for _,c in item);area=max(0,bottom-top-1)*max(0,right-left-1);boxes.append((area,top,bottom,left,right))
    boxes.sort();
    if len(boxes)!=len(rank_colors):raise ValueError('rank arity mismatch')
    output=[list(row) for row in grid]
    for color,(_,top,bottom,left,right) in zip(rank_colors,boxes):
        for r in range(top+1,bottom):
            for c in range(left+1,right):output[r][c]=color
    return as_grid(output)

@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid:Grid)->Grid:
        values=dict(self.bindings)
        if self.graph=='expand_seed_quadrants':return expand_seed_quadrants(grid,values['background'])
        if self.graph=='project_markers_to_separator':return project_markers_to_separator(grid,values['background'],values['separator'])
        if self.graph=='fill_partition_regions':return fill_partition_regions(grid,values['separator'],values['mapping'])
        if self.graph=='fill_ranked_rectangles':return fill_ranked_rectangles(grid,values['border'],values['rank_colors'])
        raise ValueError('unknown region/topology graph')

def _partition_mapping(demos:Sequence[tuple[Grid,Grid]],separator:int)->tuple[tuple[tuple[int,int],int],...]|None:
    mappings=[]
    for source,expected in demos:
        values=[]
        for coordinate,cells in partition_regions(source,separator):
            output_values={expected[r][c] for r,c in cells}
            input_values={source[r][c] for r,c in cells}
            if len(output_values)==1 and output_values!=input_values:values.append((coordinate,next(iter(output_values))))
        mappings.append(tuple(values))
    return mappings[0] if mappings and all(item==mappings[0] for item in mappings) and mappings[0] else None

def candidates(demos:Sequence[tuple[Grid,Grid]])->tuple[Candidate,...]:
    if not all(len(source)==len(expected) and len(source[0])==len(expected[0]) for source,expected in demos):return ()
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));result=set()
    for background in colors:result.add(Candidate('expand_seed_quadrants',(('background',background),)))
    for background,separator in product(colors,colors):
        if background!=separator:result.add(Candidate('project_markers_to_separator',tuple(sorted({'background':background,'separator':separator}.items()))))
    for separator in colors:
        mapping=_partition_mapping(demos,separator)
        if mapping is not None:result.add(Candidate('fill_partition_regions',(('mapping',mapping),('separator',separator))))
    output_colors=tuple(sorted({value for _,expected in demos for row in expected for value in row}))
    for border in colors:
        arities={len(components(source,border)) for source,_ in demos}
        if len(arities)==1 and 0<next(iter(arities))<=3:
            arity=next(iter(arities))
            for rank_colors in product(output_colors,repeat=arity):result.add(Candidate('fill_ranked_rectangles',(('border',border),('rank_colors',rank_colors))))
    return tuple(sorted(result,key=lambda item:(item.graph,repr(item.bindings))))

def solve(demos:Sequence[tuple[Grid,Grid]],queries:Sequence[Grid],*,budget:int=2_000_000,allowed_graphs:Sequence[str]|None=None)->dict[str,Any]:
    allowed=set(allowed_graphs) if allowed_graphs is not None else None;pool=tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed);cells=sum(len(a)*len(a[0])+len(b)*len(b[0]) for a,b in demos);work=cells*max(1,len(pool))
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
