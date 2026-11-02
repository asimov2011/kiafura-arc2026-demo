"""Typed path/connection operators derived from A-only experiences."""
from __future__ import annotations
from collections import Counter,deque
from dataclasses import dataclass
from itertools import product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def components(grid:Grid,colors:set[int],connectivity:int=4):
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value in colors};offsets=((-1,0),(1,0),(0,-1),(0,1)) if connectivity==4 else tuple((dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0));result=[]
    while pending:
        seed=min(pending);pending.remove(seed);item={seed};queue=deque([seed])
        while queue:
            r,c=queue.popleft()
            for dr,dc in offsets:
                point=r+dr,c+dc
                if point in pending:pending.remove(point);item.add(point);queue.append(point)
        result.append(frozenset(item))
    return tuple(result)
def connect_border_marker_paths(grid:Grid,background:int,marker:int)->Grid:
    height,width=len(grid),len(grid[0]);markers=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==marker];paths=set()
    for index,(r,c) in enumerate(markers):
        for rr,cc in markers[index+1:]:
            if r==rr and {c,cc}=={0,width-1}:paths.update((r,x) for x in range(width))
            elif c==cc and {r,rr}=={0,height-1}:paths.update((x,c) for x in range(height))
    if not paths:raise ValueError('border marker pair absent')
    material_colors={value for row in grid for value in row if value not in (background,marker)};activated=set()
    for item in components(grid,material_colors,8):
        if any((r+dr,c+dc) in paths for r,c in item for dr,dc in ((0,0),(-1,0),(1,0),(0,-1),(0,1))):activated.update(item)
    output=[list(row) for row in grid]
    for r,c in paths|activated:output[r][c]=marker
    return as_grid(output)
def connect_terminals_to_spine(grid:Grid,background:int,terminal:int,ink:int)->Grid:
    points=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==terminal]
    if len(points)<3:raise ValueError('terminal set absent')
    rows=Counter(r for r,_ in points);columns=Counter(c for _,c in points);output=[list(row) for row in grid]
    if max(rows.values())>=2:
        spine=min(rows,key=lambda value:(-rows[value],value));ends=[c for r,c in points if r==spine]
        for c in range(min(ends)+1,max(ends)):
            if output[spine][c]==background:output[spine][c]=ink
        for r,c in points:
            if r!=spine:
                for rr in range(min(r,spine)+1,max(r,spine)):
                    if output[rr][c]==background:output[rr][c]=ink
    else:
        spine=min(columns,key=lambda value:(-columns[value],value));ends=[r for r,c in points if c==spine]
        for r in range(min(ends)+1,max(ends)):
            if output[r][spine]==background:output[r][spine]=ink
        for r,c in points:
            if c!=spine:
                for cc in range(min(c,spine)+1,max(c,spine)):
                    if output[r][cc]==background:output[r][cc]=ink
    return as_grid(output)
def connect_domino_via_obstacle(grid:Grid,background:int,source:int,target:int,obstacle:int)->Grid:
    height,width=len(grid),len(grid[0]);sources=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==source];targets=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==target]
    if not sources or not targets:raise ValueError('domino roles absent')
    vertical=len({c for _,c in sources})==1;candidates=[]
    for direction in (-1,1):
        endpoint=(min(sources) if direction<0 else max(sources)) if vertical else (min(sources,key=lambda point:point[1]) if direction<0 else max(sources,key=lambda point:point[1]));r,c=endpoint;distance=0
        while (0<=r+direction<height) if vertical else (0<=c+direction<width):
            if vertical:r+=direction
            else:c+=direction
            if grid[r][c]!=background:
                if grid[r][c]==obstacle:candidates.append((distance,direction,(r-direction) if vertical else (c-direction)))
                break
            distance+=1
    if not candidates:raise ValueError('obstacle not reachable')
    _,direction,bridge=max(candidates);output=[list(row) for row in grid]
    if vertical:
        source_axis=sources[0][1];target_axis=targets[0][1];source_end=min(r for r,_ in sources) if direction<0 else max(r for r,_ in sources)
        for r in range(min(source_end,bridge),max(source_end,bridge)+1):output[r][source_axis]=source
        for c in range(min(source_axis,target_axis),max(source_axis,target_axis)+1):output[bridge][c]=source
        target_values=[r for r,_ in targets];target_end=max(target_values) if bridge>max(target_values) else min(target_values)
        for r in range(min(bridge,target_end),max(bridge,target_end)+1):
            if output[r][target_axis]==background:output[r][target_axis]=source
    else:
        source_axis=sources[0][0];target_axis=targets[0][0];source_end=min(c for _,c in sources) if direction<0 else max(c for _,c in sources)
        for c in range(min(source_end,bridge),max(source_end,bridge)+1):output[source_axis][c]=source
        for r in range(min(source_axis,target_axis),max(source_axis,target_axis)+1):output[r][bridge]=source
        target_values=[c for _,c in targets];target_end=max(target_values) if bridge>max(target_values) else min(target_values)
        for c in range(min(bridge,target_end),max(bridge,target_end)+1):
            if output[target_axis][c]==background:output[target_axis][c]=source
    return as_grid(output)
def connect_open_boxes(grid:Grid,background:int,box_color:int,ink:int)->Grid:
    def door(item):
        top,bottom=min(r for r,_ in item),max(r for r,_ in item);left,right=min(c for _,c in item),max(c for _,c in item)
        for side,r,c,dr,dc in (('top',top,left+1,-1,0),('bottom',bottom,left+1,1,0),('left',top+1,left,0,-1),('right',top+1,right,0,1)):
            if grid[r][c]==background:return {'side':side,'gap':(r,c),'outside':(r+dr,c+dc),'box':(top,bottom,left,right)}
        raise ValueError('box without door')
    objects=[door(item) for item in components(grid,{box_color},4)]
    if len(objects)!=2:raise ValueError('two boxes required')
    top_object,bottom_object=sorted(objects,key=lambda item:item['box'][0]);top_top,top_bottom,top_left,top_right=top_object['box'];bottom_top,_,_,_=bottom_object['box'];pair=top_object['side'],bottom_object['side']
    if pair==('right','top'):left,right=bottom_object['gap'][1],top_object['outside'][1]
    elif pair==('bottom','left'):left,right=top_object['gap'][1],bottom_object['outside'][1]
    elif pair==('top','right'):left,right=top_right+1,bottom_object['outside'][1]
    elif pair==('right','left'):left,right=bottom_object['outside'][1],top_object['outside'][1]
    else:raise ValueError('unsupported door relation')
    output=[list(row) for row in grid]
    for r in range(top_bottom+1,bottom_top):
        for c in range(left,right+1):
            if output[r][c]==background:output[r][c]=ink
    top_gap,bottom_gap=top_object['gap'],bottom_object['gap']
    if pair==('top','right'):
        output[top_gap[0]][top_gap[1]]=ink;output[bottom_gap[0]][bottom_gap[1]]=ink
        for c in range(top_gap[1],right+1):
            if output[0][c]==background:output[0][c]=ink
        for r in range(1,top_bottom+1):
            for c in range(left,right+1):
                if output[r][c]==background:output[r][c]=ink
        for r in range(bottom_top,bottom_gap[0]+1):
            if output[r][right]==background:output[r][right]=ink
    elif pair==('right','top'):
        for c in range(top_gap[1],right+1):output[top_gap[0]][c]=ink
        for r in range(top_gap[0],top_bottom+1):output[r][right]=ink
        output[bottom_gap[0]][bottom_gap[1]]=ink
    elif pair==('bottom','left'):
        output[top_gap[0]][top_gap[1]]=ink
        for r in range(bottom_top,bottom_gap[0]+1):
            for c in range(left,right+1):
                if output[r][c]==background:output[r][c]=ink
        output[bottom_gap[0]][bottom_gap[1]]=ink
    else:
        for c in range(top_gap[1],right+1):output[top_gap[0]][c]=ink
        for r in range(top_gap[0],top_bottom+1):output[r][right]=ink
        for r in range(bottom_top,bottom_gap[0]+1):
            if output[r][left]==background:output[r][left]=ink
        output[bottom_gap[0]][bottom_gap[1]]=ink
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        values=dict(self.bindings)
        if self.graph=='connect_border_marker_paths':return connect_border_marker_paths(grid,values['background'],values['marker'])
        if self.graph=='connect_terminals_to_spine':return connect_terminals_to_spine(grid,values['background'],values['terminal'],values['ink'])
        if self.graph=='connect_domino_via_obstacle':return connect_domino_via_obstacle(grid,values['background'],values['source'],values['target'],values['obstacle'])
        if self.graph=='connect_open_boxes':return connect_open_boxes(grid,values['background'],values['box_color'],values['ink'])
        raise ValueError('unknown graph')
def candidates(demos):
    if not all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos):return ()
    colors=tuple(sorted({value for pair in demos for grid in pair for row in grid for value in row}));backgrounds=tuple(sorted({Counter(value for row in source for value in row).most_common(1)[0][0] for source,_ in demos}));added=tuple(sorted({value for _,expected in demos for row in expected for value in row}-{value for source,_ in demos for row in source for value in row}));result=set()
    for background,marker in product(backgrounds,colors):
        if marker!=background:result.add(Candidate('connect_border_marker_paths',tuple(sorted({'background':background,'marker':marker}.items()))))
    for background,terminal,ink in product(backgrounds,colors,added or colors):
        if len({background,terminal,ink})==3:result.add(Candidate('connect_terminals_to_spine',tuple(sorted({'background':background,'terminal':terminal,'ink':ink}.items()))))
    for background,source,target,obstacle in product(backgrounds,colors,colors,colors):
        if len({background,source,target,obstacle})==4:result.add(Candidate('connect_domino_via_obstacle',tuple(sorted({'background':background,'source':source,'target':target,'obstacle':obstacle}.items()))))
    for background,box_color,ink in product(backgrounds,colors,added or colors):
        if len({background,box_color,ink})==3:result.add(Candidate('connect_open_boxes',tuple(sorted({'background':background,'box_color':box_color,'ink':ink}.items()))))
    return tuple(sorted(result,key=lambda item:(item.graph,repr(item.bindings))))
def solve(demos,queries,*,budget=2_000_000,allowed_graphs=None):
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
