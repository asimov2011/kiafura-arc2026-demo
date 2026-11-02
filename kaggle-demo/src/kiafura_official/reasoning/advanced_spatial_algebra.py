"""Typed spatial-control operators for the remaining A-only transform family."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def components(grid:Grid,background:int,connectivity:int=4):
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value!=background};result=[];steps=((-1,0),(1,0),(0,-1),(0,1)) if connectivity==4 else tuple((dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc)
    while pending:
        seed=pending.pop();item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for dr,dc in steps:
                neighbor=(r+dr,c+dc)
                if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
        result.append(item)
    return result
def fold_tail(grid:Grid,background:int,needle:int,support:int,erased:int)->Grid:
    height,width=len(grid),len(grid[0]);output=[list(row) for row in grid];found=0
    for empty_row in range(height):
        for empty_column in range(width):
            if grid[empty_row][empty_column]!=background:continue
            direction=None
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                r,c=empty_row-dr,empty_column-dc
                while 0<=r<height and 0<=c<width and grid[r][c]==needle:r,c=r-dr,c-dc
                if 0<=r<height and 0<=c<width and grid[r][c]==support:direction=(dr,dc);break
            if direction is None:continue
            dr,dc=direction;tail=[];r,c=empty_row+dr,empty_column+dc
            while 0<=r<height and 0<=c<width and grid[r][c]==needle:tail.append((r,c));r,c=r+dr,c+dc
            if not tail:continue
            found+=1
            for r,c in tail:output[r][c]=erased
            new_dr,new_dc=(0,1 if empty_column<(width-1)/2 else -1) if dr else (1 if empty_row<(height-1)/2 else -1,0)
            for distance in range(1,len(tail)+1):
                r,c=empty_row+new_dr*distance,empty_column+new_dc*distance
                if not (0<=r<height and 0<=c<width):raise ValueError('fold outside canvas')
                output[r][c]=needle
    if not found:raise ValueError('fold site absent')
    return as_grid(output)
def build_oriented_barrier(grid:Grid,marker:int,on:int,off:int,gap:int)->Grid:
    height,width=len(grid),len(grid[0]);markers=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==marker]
    if len(markers)!=1:raise ValueError('unique marker required')
    marker_row,marker_column=markers[0];_,side=min(((marker_row,'top'),(height-1-marker_row,'bottom'),(marker_column,'left'),(width-1-marker_column,'right')));output=[list(row) for row in grid]
    if side in ('left','right'):
        middle=width//2
        for r in range(height):
            for c in range(width):
                if width%2 and c==middle:output[r][c]=on if r%2==marker_row%2 else off;continue
                near=c<middle if side=='left' else c>=(width+1)//2;output[r][c]=on if near else off
        output[marker_row][marker_column]=marker;step=1 if side=='left' else -1;c=marker_column+step
        while 0<=c<width:
            if not (width%2 and c==middle) and output[marker_row][c]==off:output[marker_row][c]=gap
            c+=step
        if width%2==0:output[marker_row][0 if side=='right' else width-1]=off
    else:
        middle=height//2
        for r in range(height):
            if height%2 and r==middle:
                for c in range(width):output[r][c]=on if c%2==marker_column%2 else off
                continue
            near=r<middle if side=='top' else r>=(height+1)//2
            for c in range(width):output[r][c]=on if near else off
        output[marker_row][marker_column]=marker;step=1 if side=='top' else -1;r=marker_row+step
        while 0<=r<height:
            if not (height%2 and r==middle) and output[r][marker_column]==off:output[r][marker_column]=gap
            r+=step
        if height%2==0:output[0 if side=='bottom' else height-1][marker_column]=off
    return as_grid(output)
def tile_complementary_pieces(grid:Grid,background:int,output_height:int,output_width:int)->Grid:
    pieces=[]
    for item in components(grid,background,4):
        top,left=min(r for r,_ in item),min(c for _,c in item);pieces.append([(r-top,c-left,grid[r][c]) for r,c in item])
    if len(pieces)!=2:raise ValueError('two pieces required')
    placements=[]
    for piece in pieces:
        piece_height=max(r for r,_,_ in piece)+1;piece_width=max(c for _,c,_ in piece)+1
        placements.append([{(r+dr,c+dc):color for r,c,color in piece} for dr in range(output_height-piece_height+1) for dc in range(output_width-piece_width+1)])
    for first in placements[0]:
        for second in placements[1]:
            if set(first).isdisjoint(second) and len(first)+len(second)==output_height*output_width:
                merged=first|second
                if len(merged)==output_height*output_width:return as_grid([[merged[(r,c)] for c in range(output_width)] for r in range(output_height)])
    raise ValueError('complementary tiling absent')
def reflect_pivot_axes(grid:Grid,background:int)->Grid:
    counts=Counter(value for row in grid for value in row if value!=background)
    if len(counts)<2:raise ValueError('pivot roles absent')
    pivot=min(counts,key=counts.get);ink=max(counts,key=counts.get);pivots=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==pivot];row_axis=min(r for r,_ in pivots)+max(r for r,_ in pivots);column_axis=min(c for _,c in pivots)+max(c for _,c in pivots);original=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==ink];output=[list(row) for row in grid]
    for r,c in original:
        for target_row in (r,row_axis-r):
            for target_column in (c,column_axis-c):
                if 0<=target_row<len(grid) and 0<=target_column<len(grid[0]) and output[target_row][target_column]==background:output[target_row][target_column]=ink
    return as_grid(output)
def copy_anchor_templates(grid:Grid,background:int,mirror_anchor:int)->Grid:
    height,width=len(grid),len(grid[0]);items=components(grid,background,8);output=[list(row) for row in grid];found=0
    for item in items:
        colors=Counter(grid[r][c] for r,c in item)
        if len(colors)<2:continue
        anchor=min(colors,key=colors.get);source=next((r,c) for r,c in item if grid[r][c]==anchor);targets=[next(iter(candidate)) for candidate in items if len(candidate)==1 and grid[next(iter(candidate))[0]][next(iter(candidate))[1]]==anchor]
        for target_row,target_column in targets:
            found+=1
            for r,c in item:
                dr,dc=r-source[0],c-source[1]
                if anchor==mirror_anchor:dc=-dc
                destination_row,destination_column=target_row+dr,target_column+dc
                if 0<=destination_row<height and 0<=destination_column<width:output[destination_row][destination_column]=grid[r][c]
    if not found:raise ValueError('anchor copies absent')
    return as_grid(output)
@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        v=dict(self.bindings)
        if self.graph=='fold_tail':return fold_tail(grid,v['background'],v['needle'],v['support'],v['erased'])
        if self.graph=='build_oriented_barrier':return build_oriented_barrier(grid,v['marker'],v['on'],v['off'],v['gap'])
        if self.graph=='tile_complementary_pieces':return tile_complementary_pieces(grid,v['background'],v['output_height'],v['output_width'])
        if self.graph=='reflect_pivot_axes':return reflect_pivot_axes(grid,v['background'])
        if self.graph=='copy_anchor_templates':return copy_anchor_templates(grid,v['background'],v['mirror_anchor'])
        raise ValueError('unknown graph')
def candidates(demos):
    common=None;union=set();shapes={(len(expected),len(expected[0])) for _,expected in demos};result=set()
    for source,expected in demos:
        palette={value for row in source for value in row};common=palette if common is None else common&palette;union.update(palette);union.update(value for row in expected for value in row)
    common_colors=tuple(sorted(common or ()));all_colors=tuple(sorted(union))
    for background in common_colors:
        result.add(Candidate('reflect_pivot_axes',(('background',background),)))
        for mirror_anchor in all_colors:
            if mirror_anchor!=background:result.add(Candidate('copy_anchor_templates',(('background',background),('mirror_anchor',mirror_anchor))))
        if len(shapes)==1:
            output_height,output_width=next(iter(shapes));result.add(Candidate('tile_complementary_pieces',(('background',background),('output_height',output_height),('output_width',output_width))))
        for needle,support in permutations([value for value in common_colors if value!=background],2):
            for erased in all_colors:result.add(Candidate('fold_tail',(('background',background),('erased',erased),('needle',needle),('support',support))))
    for marker in common_colors:
        for on,off,gap in permutations([value for value in all_colors if value!=marker],3):result.add(Candidate('build_oriented_barrier',(('gap',gap),('marker',marker),('off',off),('on',on))))
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
