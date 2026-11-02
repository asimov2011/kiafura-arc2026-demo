"""Typed spatial transform operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter,deque
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def arrange_l_trominoes(grid:Grid,background:int,size:int)->Grid:
    output=[[background]*size for _ in range(size)];colors=sorted({value for row in grid for value in row if value!=background})
    for color in colors:
        pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color}
        while pending:
            seed=min(pending);pending.remove(seed);item={seed};queue=deque([seed])
            while queue:
                r,c=queue.popleft()
                for point in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if point in pending:pending.remove(point);item.add(point);queue.append(point)
            if len(item)!=3:continue
            top,left=min(r for r,_ in item),min(c for _,c in item);shape={(r-top,c-left) for r,c in item};missing=next(point for point in ((0,0),(0,1),(1,0),(1,1)) if point not in shape);origin_r=0 if missing[0]==1 else size-2;origin_c=0 if missing[1]==1 else size-2
            for r,c in shape:output[origin_r+r][origin_c+c]=color
    return as_grid(output)
def modal_background(grid:Grid)->int:
    return Counter(value for row in grid for value in row).most_common(1)[0][0]
def reflect_using_guide(grid:Grid)->Grid:
    background=modal_background(grid)
    height,width=len(grid),len(grid[0]);colors={value for row in grid for value in row if value!=background}
    if len(colors)!=2:raise ValueError('two roles required')
    candidates=[]
    for main in colors:
        ink=next(iter(colors-{main}));source={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==main};guide={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==ink}
        for axis,limit in (('row',2*height-1),('column',2*width-1)):
            for coordinate in range(limit):
                mapped={(coordinate-r,c) if axis=='row' else (r,coordinate-c) for r,c in source};inside={(r,c) for r,c in mapped if 0<=r<height and 0<=c<width};overlap=len(inside&guide);conflict=len(inside&source);novel=len(inside-source-guide)
                if overlap:candidates.append((overlap,-conflict,novel,len(inside),coordinate,ink,inside))
    if not candidates:raise ValueError('guide absent')
    *_,ink,mapped=max(candidates,key=lambda item:item[:5]);output=[list(row) for row in grid]
    for r,c in mapped:
        if output[r][c]==background:output[r][c]=ink
    return as_grid(output)
def reflect_shape_from_marker(grid:Grid,background:int,shape_color:int,marker_color:int)->Grid:
    marker={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==marker_color};shape={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==shape_color}
    if not marker or not shape:raise ValueError('roles absent')
    directions=((-1,0),(0,1),(1,0),(0,-1));junction=max(marker,key=lambda point:sum((point[0]+dr,point[1]+dc) in marker for dr,dc in directions));lengths=[]
    for dr,dc in directions:
        length=0;r,c=junction
        while (r+dr,c+dc) in marker:length+=1;r+=dr;c+=dc
        lengths.append(length)
    direction=directions[max(range(4),key=lambda index:lengths[index])];dr,dc=-direction[1],direction[0]
    if dc>0:axis=2*max(c for _,c in shape)+1;reflected={(r,axis-c) for r,c in shape}
    elif dc<0:axis=2*min(c for _,c in shape)-1;reflected={(r,axis-c) for r,c in shape}
    elif dr>0:axis=2*max(r for r,_ in shape)+1;reflected={(axis-r,c) for r,c in shape}
    else:axis=2*min(r for r,_ in shape)-1;reflected={(axis-r,c) for r,c in shape}
    output=[[background if value==marker_color else value for value in row] for row in grid]
    for r,c in reflected:
        if 0<=r<len(grid) and 0<=c<len(grid[0]):output[r][c]=shape_color
    return as_grid(output)
def translate_figure_to_edge(grid:Grid)->Grid:
    background=modal_background(grid);edge_by_color=((3,'left'),(4,'bottom'),(6,'top'),(8,'right'))
    points=[(r,c,value) for r,row in enumerate(grid) for c,value in enumerate(row) if value!=background]
    if not points:raise ValueError('figure absent')
    colors={value for _,_,value in points};mapping=dict(edge_by_color)
    if len(colors)!=1 or next(iter(colors)) not in mapping:raise ValueError('edge binding absent')
    color=next(iter(colors));top,bottom=min(r for r,_,_ in points),max(r for r,_,_ in points);left,right=min(c for _,c,_ in points),max(c for _,c,_ in points);shape=tuple(tuple(grid[r][left:right+1]) for r in range(top,bottom+1));edge=mapping[color];origin_r=0 if edge=='top' else len(grid)-len(shape) if edge=='bottom' else top;origin_c=0 if edge=='left' else len(grid[0])-len(shape[0]) if edge=='right' else left;output=[[background]*len(grid[0]) for _ in grid]
    for r,row in enumerate(shape):
        for c,value in enumerate(row):output[origin_r+r][origin_c+c]=value
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='arrange_l_trominoes':return arrange_l_trominoes(grid,v['background'],v['size'])
        if self.graph=='reflect_using_guide':return reflect_using_guide(grid)
        if self.graph=='reflect_shape_from_marker':return reflect_shape_from_marker(grid,v['background'],v['shape_color'],v['marker_color'])
        if self.graph=='translate_figure_to_edge':return translate_figure_to_edge(grid)
        raise ValueError('unknown graph')
def _edge_mapping(demos,background):
    mapping={}
    for source,expected in demos:
        colors={value for row in source for value in row if value!=background}
        if len(colors)!=1:return None
        color=next(iter(colors));cells=[(r,c) for r,row in enumerate(expected) for c,value in enumerate(row) if value==color]
        if not cells:return None
        edges=[edge for edge,condition in (('top',min(r for r,_ in cells)==0),('bottom',max(r for r,_ in cells)==len(expected)-1),('left',min(c for _,c in cells)==0),('right',max(c for _,c in cells)==len(expected[0])-1)) if condition]
        if len(edges)!=1:return None
        if color in mapping and mapping[color]!=edges[0]:return None
        mapping[color]=edges[0]
    return tuple(sorted(mapping.items())) if mapping else None
def candidates(demos):
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));backgrounds=tuple(sorted({Counter(value for row in source for value in row).most_common(1)[0][0] for source,_ in demos}));same=all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos);result=set()
    if not same:
        shapes={(len(expected),len(expected[0])) for _,expected in demos}
        if len(shapes)==1 and next(iter(shapes))[0]==next(iter(shapes))[1]:
            size=next(iter(shapes))[0]
            for background in backgrounds:result.add(Candidate('arrange_l_trominoes',(('background',background),('size',size))))
        return tuple(sorted(result,key=lambda item:(item.graph,repr(item.bindings))))
    result.add(Candidate('reflect_using_guide',()))
    if all(_edge_mapping(demos,background) is not None for background in backgrounds):result.add(Candidate('translate_figure_to_edge',()))
    for background in backgrounds:
        for shape_color,marker_color in product(colors,colors):
            if len({background,shape_color,marker_color})==3:result.add(Candidate('reflect_shape_from_marker',tuple(sorted({'background':background,'marker_color':marker_color,'shape_color':shape_color}.items()))))
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
