"""Typed frame extraction, completion, and repair operators from A-only evidence."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def background_of(grid:Grid)->int:return Counter(value for row in grid for value in row).most_common(1)[0][0]

def complete_encoded_frames(grid:Grid,frame_color:int,marker_color:int)->Grid:
    height,width=len(grid),len(grid[0]);frame=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==frame_color]
    if not frame:raise ValueError('frame absent')
    top,bottom=min(r for r,_ in frame),max(r for r,_ in frame);left,right=min(c for _,c in frame),max(c for _,c in frame);frame_height,frame_width=bottom-top+1,right-left+1;markers={(r-top,c-left) for r in range(top,bottom+1) for c in range(left,right+1) if grid[r][c]==marker_color}
    if not markers:raise ValueError('markers absent')
    forms=set()
    for reflected in (False,True):
        form_height,form_width=frame_height,frame_width;form_markers={(r,frame_width-1-c if reflected else c) for r,c in markers}
        for _ in range(4):
            forms.add((form_height,form_width,tuple(sorted(form_markers))));form_markers={(c,form_height-1-r) for r,c in form_markers};form_height,form_width=form_width,form_height
    output=[list(row) for row in grid];background=background_of(grid)
    for form_height,form_width,marker_tuple in forms:
        form_markers=set(marker_tuple)
        for origin_row in range(-form_height+1,height):
            for origin_column in range(-form_width+1,width):
                if not all(0<=origin_row+r<height and 0<=origin_column+c<width for r,c in form_markers):continue
                visible={(r-origin_row,c-origin_column) for r in range(max(0,origin_row),min(height,origin_row+form_height)) for c in range(max(0,origin_column),min(width,origin_column+form_width)) if grid[r][c]==marker_color}
                if visible!=form_markers:continue
                for r in range(max(0,origin_row),min(height,origin_row+form_height)):
                    for c in range(max(0,origin_column),min(width,origin_column+form_width)):
                        if output[r][c]==background:output[r][c]=frame_color
    return as_grid(output)

def complete_mirrored_tiles(grid:Grid,size:int)->Grid:
    height,width=len(grid),len(grid[0]);background=background_of(grid);base=None
    for top in range(height-size+1):
        for left in range(width-size+1):
            block=[list(row[left:left+size]) for row in grid[top:top+size]];colors={value for row in block for value in row}
            if background not in colors and len(colors)==2:base=(top,left,block,colors);break
        if base is not None:break
    if base is None:raise ValueError('canonical tile absent')
    top,left,pattern,colors=base;output=[list(row) for row in grid];variants=((0,0,pattern),(0,size,[row[::-1] for row in pattern]),(size,0,pattern[::-1]),(size,size,[row[::-1] for row in pattern[::-1]]))
    for row_offset,column_offset,variant in variants[1:]:
        mapping={}
        for r in range(size):
            for c in range(size):
                target_row,target_column=top+row_offset+r,left+column_offset+c
                if target_row<height and target_column<width and grid[target_row][target_column]!=background:mapping[variant[r][c]]=grid[target_row][target_column]
        for source_color in colors:mapping.setdefault(source_color,background)
        for r in range(size):
            for c in range(size):
                target_row,target_column=top+row_offset+r,left+column_offset+c
                if target_row<height and target_column<width:output[target_row][target_column]=mapping[variant[r][c]]
    return as_grid(output)

def reconstruct_frame_tails(grid:Grid)->Grid:
    height,width=len(grid),len(grid[0]);background=background_of(grid);counts=Counter(value for row in grid for value in row if value!=background);candidates=[]
    if len(counts)!=2:raise ValueError('two foreground roles required')
    for color in counts:
        points=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color];top,bottom=min(r for r,_ in points),max(r for r,_ in points);left,right=min(c for _,c in points),max(c for _,c in points);perimeter=2*(bottom-top+right-left);coverage=len(points)/perimeter if perimeter else 0
        if bottom-top<height-1 and right-left<width-1:candidates.append((coverage,color))
    if not candidates:raise ValueError('frame role absent')
    frame_color=max(candidates)[1];line_color=next(color for color in counts if color!=frame_color);frame_points=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==frame_color];top,bottom=min(r for r,_ in frame_points),max(r for r,_ in frame_points);left,right=min(c for _,c in frame_points),max(c for _,c in frame_points);line_points=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==line_color];minimum_row,maximum_row=min(r for r,_ in line_points),max(r for r,_ in line_points);minimum_column,maximum_column=min(c for _,c in line_points),max(c for _,c in line_points);output=[[background]*width for _ in range(height)]
    for c in range(left,right+1):output[top][c]=frame_color;output[bottom][c]=frame_color
    for r in range(top,bottom+1):output[r][left]=frame_color;output[r][right]=frame_color
    for r in range(minimum_row,maximum_row+1):
        if r<top or r>bottom:output[r][right]=line_color
    for c in range(minimum_column,maximum_column+1):
        if c<left or c>right:output[top][c]=line_color
    return as_grid(output)

def repair_homologous_frames(grid:Grid,background:int)->Grid:
    height,width=len(grid),len(grid[0]);references=[]
    for top in range(1,height-1):
        for bottom in range(top,height-1):
            for left in range(1,width-1):
                for right in range(left,width-1):
                    if not all(grid[r][c]==background for r in range(top,bottom+1) for c in range(left,right+1)):continue
                    border=[grid[top-1][c] for c in range(left-1,right+2)]+[grid[bottom+1][c] for c in range(left-1,right+2)]+[grid[r][left-1] for r in range(top,bottom+1)]+[grid[r][right+1] for r in range(top,bottom+1)]
                    if len(set(border))==1 and border[0]!=background:references.append(((bottom-top+1)*(right-left+1),border[0],bottom-top+1))
    if not references:raise ValueError('reference frame absent')
    _,color,inner_height=max(references);candidates=[]
    for top in range(height-inner_height+1):
        zero_columns=[all(grid[r][c]==background for r in range(top,top+inner_height)) for c in range(width)];column=0
        while column<width:
            if not zero_columns[column]:column+=1;continue
            left=column
            while column<width and zero_columns[column]:column+=1
            right=column-1
            if right-left+1>=2 and top>0 and top+inner_height<height and left>0 and right+1<width:candidates.append((right-left+1,top,left,right))
    if not candidates:raise ValueError('homologous cavity absent')
    maximum_width=max(candidate[0] for candidate in candidates);output=[list(row) for row in grid]
    for candidate_width,top,left,right in candidates:
        if candidate_width!=maximum_width:continue
        bottom=top+inner_height-1
        for c in range(left-1,right+2):output[top-1][c]=color;output[bottom+1][c]=color
        for r in range(top,bottom+1):output[r][left-1]=color;output[r][right+1]=color
    return as_grid(output)

@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        values=dict(self.bindings)
        if self.graph=='complete_encoded_frames':return complete_encoded_frames(grid,values['frame_color'],values['marker_color'])
        if self.graph=='complete_mirrored_tiles':return complete_mirrored_tiles(grid,values['size'])
        if self.graph=='reconstruct_frame_tails':return reconstruct_frame_tails(grid)
        if self.graph=='repair_homologous_frames':return repair_homologous_frames(grid,values['background'])
        raise ValueError('unknown graph')
def candidates(demos):
    if not all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos):return ()
    common=None;common_palette=None;minimum=min(min(len(a),len(a[0])) for a,_ in demos);result={Candidate('reconstruct_frame_tails',())}
    for source,_ in demos:
        background=background_of(source);palette={value for row in source for value in row};roles=palette-{background};common=roles if common is None else common&roles;common_palette=palette if common_palette is None else common_palette&palette
    for frame_color,marker_color in permutations(sorted(common or ()),2):result.add(Candidate('complete_encoded_frames',(('frame_color',frame_color),('marker_color',marker_color))))
    for background in sorted(common_palette or ()):result.add(Candidate('repair_homologous_frames',(('background',background),)))
    for size in range(2,minimum//2+1):result.add(Candidate('complete_mirrored_tiles',(('size',size),)))
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
