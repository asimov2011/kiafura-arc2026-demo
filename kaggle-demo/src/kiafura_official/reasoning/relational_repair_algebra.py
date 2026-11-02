"""Typed relational-control and repair operators derived from A-only evidence."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import permutations,product
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
RECTANGLE_COMPLETION_MAX_CELLS=256
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def rectangle_completion_applicable(grid:Grid)->bool:
    """Keep the combinatorial operator inside its frozen A-source envelope."""
    return bool(grid and grid[0] and len(grid)*len(grid[0])<=RECTANGLE_COMPLETION_MAX_CELLS)
def contour_turns(grid:Grid,background:int,outer_color:int,inner_color:int)->Grid:
    roles={value for row in grid for value in row if value!=background}
    if len(roles)!=1:raise ValueError('single contour role required')
    path_color=next(iter(roles));output=[list(row) for row in grid];pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==path_color};found=0
    while pending:
        seed=min(pending);pending.remove(seed);component={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in pending:pending.remove(neighbor);component.add(neighbor);stack.append(neighbor)
        order=[min(component)];previous=None
        while len(order)<len(component):
            r,c=order[-1];neighbors=sorted(neighbor for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if neighbor in component and neighbor!=previous)
            if not neighbors:raise ValueError('open contour')
            following=neighbors[0]
            if following==order[0] and len(neighbors)>1 and len(order)<len(component):following=neighbors[1]
            previous=order[-1];order.append(following)
        area=sum(order[i][1]*order[(i+1)%len(order)][0]-order[(i+1)%len(order)][1]*order[i][0] for i in range(len(order)))
        for i,(r,c) in enumerate(order):
            previous_row,previous_column=order[i-1];next_row,next_column=order[(i+1)%len(order)];turn=(r-previous_row)*(next_column-c)-(c-previous_column)*(next_row-r)
            if turn:output[r][c]=outer_color if turn*area<0 else inner_color;found+=1
    if not found:raise ValueError('turns absent')
    return as_grid(output)
def expand_key_sequences(grid:Grid,background:int,minimum_length:int=3)->Grid:
    sequences=[];protected=set()
    for r,values in enumerate(grid):
        start=None
        for c,value in enumerate((*values,background)):
            if value!=background and start is None:start=c
            elif value==background and start is not None:
                if c-start>=minimum_length:sequences.append(values[start:c]);protected.update((r,index) for index in range(start,c))
                start=None
    if not sequences:raise ValueError('key sequence absent')
    output=[list(row) for row in grid]
    for r,row in enumerate(grid):
        for c,value in enumerate(row):
            if value==background or (r,c) in protected:continue
            matches=[sequence for sequence in sequences if value in sequence]
            if not matches:raise ValueError('marker without key')
            sequence=matches[0];origin=c-sequence.index(value)
            for index,item in enumerate(sequence):
                target=origin+index
                if 0<=target<len(grid[0]):output[r][target]=item
    return as_grid(output)
def complete_overlapping_rectangles(grid:Grid,foreground_color:int,anomaly_color:int)->Grid:
    if not rectangle_completion_applicable(grid):raise ValueError('rectangle completion outside resource envelope')
    height,width=len(grid),len(grid[0]);foreground={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==foreground_color};foreground_mask=sum(1<<(r*width+c) for r,c in foreground);candidates=[]
    for top in range(height):
        for bottom in range(top+2,height):
            rectangle_height=bottom-top+1
            for left in range(width):
                for right in range(left+2,width):
                    rectangle_width=right-left+1
                    if max(rectangle_height,rectangle_width)<6:continue
                    row_mask=(1<<rectangle_width)-1;mask=sum(row_mask<<(r*width+left) for r in range(top,bottom+1));area=rectangle_height*rectangle_width;support=(mask&foreground_mask).bit_count()
                    if support*10>=area*7:candidates.append((mask,(top,bottom,left,right)))
    best=None
    for index,(first_mask,first_rectangle) in enumerate(candidates):
        for second_mask,second_rectangle in candidates[index:]:
            union=first_mask|second_mask;key=((foreground_mask^union).bit_count(),-union.bit_count(),first_rectangle,second_rectangle)
            if best is None or key<best[0]:best=(key,first_rectangle,second_rectangle)
    if best is None:raise ValueError('rectangle pair absent')
    base=set()
    for top,bottom,left,right in best[1:]:base.update((r,c) for r in range(top,bottom+1) for c in range(left,right+1))
    output=[list(row) for row in grid]
    def runs(values):
        result=[]
        for value in values:
            if not result or value>result[-1][1]+1:result.append((value,value))
            else:result[-1]=(result[-1][0],value)
        return result
    for r in range(height):
        for left,right in runs(sorted(c for current,c in base if current==r)):
            if (r,left) not in foreground and right+1<width:output[r][right+1]=foreground_color
            if (r,right) not in foreground and left>0:output[r][left-1]=foreground_color
            if left>0 and (r,left-1) in foreground:output[r][right]=anomaly_color
            if right+1<width and (r,right+1) in foreground:output[r][left]=anomaly_color
    for c in range(width):
        for top,bottom in runs(sorted(r for r,current in base if current==c)):
            if (top,c) not in foreground and bottom+1<height:output[bottom+1][c]=foreground_color
            if (bottom,c) not in foreground and top>0:output[top-1][c]=foreground_color
            if top>0 and (top-1,c) in foreground:output[bottom][c]=anomaly_color
            if bottom+1<height and (bottom+1,c) in foreground:output[top][c]=anomaly_color
    return as_grid(output)
def occlusion_cost(grid:Grid,occluder:int)->int:
    height,width=len(grid),len(grid[0]);pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==occluder};cost=height*width
    while pending:
        seed=pending.pop();item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
        colors={grid[nr][nc] for r,c in item for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if 0<=nr<height and 0<=nc<width and grid[nr][nc]!=occluder};cost+=len(colors)**len(item)
    return cost
def repair_occluded_analogues(grid:Grid,occluder:int)->Grid:
    height,width=len(grid),len(grid[0]);output=[list(row) for row in grid];pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==occluder};items=[]
    while pending:
        seed=pending.pop();item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
        items.append(sorted(item))
    if not items:raise ValueError('occlusion absent')
    for item in items:
        item_set=set(item);colors=sorted({grid[nr][nc] for r,c in item for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if 0<=nr<height and 0<=nc<width and grid[nr][nc]!=occluder})
        if not colors or len(colors)**len(item)>200_000:raise ValueError('occlusion budget')
        exemplar={}
        for r,c in item:
            for color in colors:
                costs=[]
                for other_row in range(height):
                    if other_row!=r and grid[other_row][c]==color:
                        visible=[x for x in range(width) if grid[r][x]!=occluder and grid[other_row][x]!=occluder];costs.append(sum(grid[r][x]!=grid[other_row][x] for x in visible))
                for other_column in range(width):
                    if other_column!=c and grid[r][other_column]==color:
                        visible=[x for x in range(height) if grid[x][c]!=occluder and grid[x][other_column]!=occluder];costs.append(sum(grid[x][c]!=grid[x][other_column] for x in visible))
                exemplar[r,c,color]=min(costs) if costs else height+width
        alternatives=[]
        for values in product(colors,repeat=len(item)):
            assignment=dict(zip(item,values));analog=sum(exemplar[r,c,assignment[r,c]] for r,c in item);boundaries=0
            for r,c in item:
                for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    nr,nc=neighbor
                    if not (0<=nr<height and 0<=nc<width):continue
                    if neighbor in item_set:
                        if (r,c)<neighbor:boundaries+=assignment[r,c]!=assignment[neighbor]
                    else:boundaries+=assignment[r,c]!=grid[nr][nc]
            alternatives.append((analog,boundaries,values))
        pareto=[candidate for candidate in alternatives if not any(other[0]<=candidate[0] and other[1]<=candidate[1] and other[:2]!=candidate[:2] for other in alternatives)];degree=max(sum(neighbor in item_set for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))) for r,c in item);coherent=[candidate for candidate in pareto if candidate[1]<=degree];selected=min(coherent,key=lambda candidate:(candidate[1],candidate[0],candidate[2]))[2] if coherent else min(pareto,key=lambda candidate:(candidate[0],candidate[1],candidate[2]))[2]
        for (r,c),color in zip(item,selected):output[r][c]=color
    return as_grid(output)
def propagate_boundary_dominance(grid:Grid,background:int,marker:int,min_span:int=4)->Grid:
    height,width=len(grid),len(grid[0]);barriers=set()
    for r in range(height):
        c=0
        while c<width:
            if grid[r][c]==background:c+=1;continue
            start=c
            while c<width and grid[r][c]!=background:c+=1
            if c-start>=min_span:barriers.update((r,x) for x in range(start,c))
    for c in range(width):
        r=0
        while r<height:
            if grid[r][c]==background:r+=1;continue
            start=r
            while r<height and grid[r][c]!=background:r+=1
            if r-start>=min_span:barriers.update((x,c) for x in range(start,r))
    pending={(r,c) for r in range(height) for c in range(width) if (r,c) not in barriers};output=[list(row) for row in grid];colored=[(r,c,grid[r][c]) for r,c in barriers if grid[r][c] not in (background,marker)];changed=0
    while pending:
        seed=pending.pop();region={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in pending:pending.remove(neighbor);region.add(neighbor);stack.append(neighbor)
        markers=[point for point in region if grid[point[0]][point[1]]==marker]
        if not markers:continue
        boundary=Counter(color for br,bc,color in colored if any(max(abs(br-r),abs(bc-c))<=1 for r,c in region))
        if not boundary:raise ValueError('boundary evidence absent')
        selected=min(boundary,key=lambda color:(-boundary[color],color))
        for r,c in markers:output[r][c]=selected;changed+=1
    if not changed:raise ValueError('markers absent')
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def applicable(self,grid):return self.graph!='complete_overlapping_rectangles' or rectangle_completion_applicable(grid)
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='contour_turns':return contour_turns(grid,v['background'],v['outer_color'],v['inner_color'])
        if self.graph=='expand_key_sequences':return expand_key_sequences(grid,v['background'])
        if self.graph=='complete_overlapping_rectangles':return complete_overlapping_rectangles(grid,v['foreground_color'],v['anomaly_color'])
        if self.graph=='repair_occluded_analogues':return repair_occluded_analogues(grid,v['occluder'])
        if self.graph=='propagate_boundary_dominance':return propagate_boundary_dominance(grid,v['background'],v['marker'])
        raise ValueError('unknown graph')
    def cost(self,grid):return occlusion_cost(grid,dict(self.bindings)['occluder']) if self.graph=='repair_occluded_analogues' else len(grid)*len(grid[0])
def candidates(demos):
    if not all(len(source)==len(expected) and len(source[0])==len(expected[0]) for source,expected in demos):return ()
    common=None;source_union=set();output_union=set();changed_inputs=set();disappeared=None;result=set();two_role_rectangles=True
    for source,expected in demos:
        palette={value for row in source for value in row};expected_palette={value for row in expected for value in row};lost=palette-expected_palette;disappeared=lost if disappeared is None else disappeared&lost;two_role_rectangles &= len(palette)==2 and expected_palette<=palette;common=palette if common is None else common&palette;source_union.update(palette);output_union.update(expected_palette);changed_inputs.update(source[r][c] for r in range(len(source)) for c in range(len(source[0])) if source[r][c]!=expected[r][c])
    common_colors=tuple(sorted(common or ()));new_colors=tuple(sorted(output_union-source_union))
    for background in common_colors:
        result.add(Candidate('expand_key_sequences',(('background',background),)))
        for marker in sorted(changed_inputs):
            if marker!=background:result.add(Candidate('propagate_boundary_dominance',(('background',background),('marker',marker))))
        for outer_color,inner_color in permutations(new_colors,2):result.add(Candidate('contour_turns',(('background',background),('inner_color',inner_color),('outer_color',outer_color))))
    for occluder in sorted(disappeared or ()):result.add(Candidate('repair_occluded_analogues',(('occluder',occluder),)))
    if two_role_rectangles and all(rectangle_completion_applicable(source) for source,_ in demos):
        for foreground,anomaly in permutations(common_colors,2):result.add(Candidate('complete_overlapping_rectangles',(('anomaly_color',anomaly),('foreground_color',foreground))))
    return tuple(sorted(result,key=lambda item:(item.graph,repr(item.bindings))))
def solve(demos,queries,*,budget=2_000_000,allowed_graphs=None):
    allowed=set(allowed_graphs) if allowed_graphs is not None else None;pool=tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed);work=sum(candidate.cost(source)+candidate.cost(expected) for candidate in pool for source,expected in demos)
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
