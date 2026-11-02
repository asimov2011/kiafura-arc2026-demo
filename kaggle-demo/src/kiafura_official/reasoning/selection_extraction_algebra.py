"""Typed selection/extraction operators derived from A-only experiences."""

from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

Grid=tuple[tuple[int,...],...]
Component=frozenset[tuple[int,int]]

def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)

def components(grid:Grid,color:int,connectivity:int)->tuple[Component,...]:
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color};offsets=((-1,0),(1,0),(0,-1),(0,1))
    if connectivity==8:offsets=tuple((dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0))
    result=[]
    while pending:
        seed=min(pending);pending.remove(seed);item={seed};queue=deque([seed])
        while queue:
            r,c=queue.popleft()
            for dr,dc in offsets:
                point=r+dr,c+dc
                if point in pending:pending.remove(point);item.add(point);queue.append(point)
        result.append(frozenset(item))
    return tuple(sorted(result,key=lambda item:(-len(item),min(item))))

def retain_key_internal_color(grid:Grid,background:int,marker:int)->Grid:
    marker_components=components(grid,marker,4)
    if not marker_components:raise ValueError('marker absent')
    key_cells=marker_components[0];top=min(r for r,_ in key_cells);bottom=max(r for r,_ in key_cells);left=min(c for _,c in key_cells);right=max(c for _,c in key_cells)
    values=[grid[r][c] for r in range(top,bottom+1) for c in range(left,right+1) if grid[r][c] not in (background,marker)]
    if not values:raise ValueError('key value absent')
    key=min(Counter(values),key=lambda value:(-Counter(values)[value],value));output=[list(row) for row in grid]
    for r,row in enumerate(grid):
        for c,value in enumerate(row):
            if value==key and not(top<=r<=bottom and left<=c<=right):output[r][c]=background
    return as_grid(output)

def retain_dominant_lattice(grid:Grid,background:int,minimum_neighbors:int)->Grid:
    counts=Counter(value for row in grid for value in row if value!=background)
    if not counts:raise ValueError('empty foreground')
    main=min(counts,key=lambda value:(-counts[value],value));height,width=len(grid),len(grid[0]);output=[[background]*width for _ in range(height)]
    for r,row in enumerate(grid):
        for c,value in enumerate(row):
            neighbors=sum(0<=r+dr<height and 0<=c+dc<width and grid[r+dr][c+dc]==main for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)))
            if value==main or (value!=background and neighbors>=minimum_neighbors):output[r][c]=main
    return as_grid(output)

def remove_bbox_contained_components(grid:Grid,background:int,connectivity:int,minimum_frame_size:int)->Grid:
    palette=sorted({value for row in grid for value in row if value!=background});objects=[(color,item) for color in palette for item in components(grid,color,connectivity)];output=[list(row) for row in grid]
    for color,item in objects:
        if len(item)<minimum_frame_size:continue
        top=min(r for r,_ in item);bottom=max(r for r,_ in item);left=min(c for _,c in item);right=max(c for _,c in item)
        for other_color,other in objects:
            if other_color!=color and all(top<=r<=bottom and left<=c<=right for r,c in other):
                for r,c in other:output[r][c]=background
    return as_grid(output)

def extract_ordered_lines(grid:Grid,background:int,ignored:int)->Grid:
    height,width=len(grid),len(grid[0]);row_lines=[]
    for r,row in enumerate(grid):
        colors={value for value in row if value not in (background,ignored)}
        if len(colors)==1:
            color=next(iter(colors))
            if row.count(color)>=max(2,width//3):row_lines.append((r,color))
    column_lines=[]
    for c in range(width):
        column=[grid[r][c] for r in range(height)];colors={value for value in column if value not in (background,ignored)}
        if len(colors)==1:
            color=next(iter(colors))
            if column.count(color)>=max(2,height//3):column_lines.append((c,color))
    lines=row_lines if len(row_lines)>=len(column_lines) else column_lines
    if not lines:raise ValueError('ordered lines absent')
    colors=[color for _,color in lines];side=len(colors)
    return as_grid([[color]*side for color in colors] if lines is row_lines else [colors[:] for _ in range(side)])

def _repeated_motif(points:set[tuple[int,int]],side:int)->tuple[int,frozenset[tuple[int,int]]|None]:
    best=(1,None)
    for origin_r in range(min(r for r,_ in points),max(r for r,_ in points)+1):
        for origin_c in range(min(c for _,c in points),max(c for _,c in points)+1):
            window={(r-origin_r,c-origin_c) for r,c in points if origin_r<=r<origin_r+side and origin_c<=c<origin_c+side}
            if not window:continue
            if max(r for r,_ in window)-min(r for r,_ in window)!=side-1 or max(c for _,c in window)-min(c for _,c in window)!=side-1:continue
            min_r=min(r for r,_ in window);min_c=min(c for _,c in window);shape=frozenset((r-min_r,c-min_c) for r,c in window);copies=[]
            for anchor_r,anchor_c in points:
                for shape_r,shape_c in shape:
                    top,left=anchor_r-shape_r,anchor_c-shape_c;placed={(top+r,left+c) for r,c in shape}
                    if placed<=points and placed not in copies:copies.append(placed)
            remaining=set(points);count=0
            for placed in sorted(copies,key=min):
                if placed<=remaining:remaining-=placed;count+=1
            if not remaining and count>best[0]:best=(count,shape)
    return best

def extract_exact_repeated_motif(grid:Grid,background:int,side:int)->Grid:
    winner=None
    for color in sorted({value for row in grid for value in row if value!=background}):
        points={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color};copies,shape=_repeated_motif(points,side)
        if shape is not None and (winner is None or copies>winner[0]):winner=(copies,color,shape)
    if winner is None:raise ValueError('repeated motif absent')
    _,color,shape=winner;return as_grid([[color if (r,c) in shape else background for c in range(side)] for r in range(side)])

@dataclass(frozen=True)
class Candidate:
    graph:str
    bindings:tuple[tuple[str,Any],...]
    def execute(self,grid:Grid)->Grid:
        values=dict(self.bindings)
        if self.graph=='retain_key_internal_color':return retain_key_internal_color(grid,values['background'],values['marker'])
        if self.graph=='retain_dominant_lattice':return retain_dominant_lattice(grid,values['background'],values['minimum_neighbors'])
        if self.graph=='remove_bbox_contained_components':return remove_bbox_contained_components(grid,values['background'],values['connectivity'],values['minimum_frame_size'])
        if self.graph=='extract_ordered_lines':return extract_ordered_lines(grid,values['background'],values['ignored'])
        if self.graph=='extract_exact_repeated_motif':return extract_exact_repeated_motif(grid,values['background'],values['side'])
        raise ValueError('unknown selection/extraction graph')

def candidates(demos:Sequence[tuple[Grid,Grid]])->tuple[Candidate,...]:
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));same=all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos);result=set()
    modal_roles={min(Counter(value for row in source for value in row),key=lambda value:(-Counter(value for row in source for value in row)[value],value)) for source,_ in demos}
    backgrounds=tuple(sorted(modal_roles))
    if same:
        for background,marker in product(backgrounds,colors):
            if marker!=background:result.add(Candidate('retain_key_internal_color',tuple(sorted({'background':background,'marker':marker}.items()))))
        for background,neighbors in product(backgrounds,range(1,5)):result.add(Candidate('retain_dominant_lattice',tuple(sorted({'background':background,'minimum_neighbors':neighbors}.items()))))
        for background,connectivity,size in product(backgrounds,(4,8),range(2,6)):result.add(Candidate('remove_bbox_contained_components',tuple(sorted({'background':background,'connectivity':connectivity,'minimum_frame_size':size}.items()))))
    else:
        outputs=tuple(expected for _,expected in demos)
        ordered_shape=all(all(len(set(row))==1 for row in expected) or all(len({expected[r][c] for r in range(len(expected))})==1 for c in range(len(expected[0]))) for expected in outputs)
        if ordered_shape:
            for background,ignored in product(backgrounds,colors):
                if ignored!=background:result.add(Candidate('extract_ordered_lines',tuple(sorted({'background':background,'ignored':ignored}.items()))))
        output_shapes={(len(expected),len(expected[0])) for _,expected in demos}
        if not ordered_shape and len(output_shapes)==1:
            height,width=next(iter(output_shapes))
            if height==width:
                for background in backgrounds:result.add(Candidate('extract_exact_repeated_motif',tuple(sorted({'background':background,'side':height}.items()))))
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
