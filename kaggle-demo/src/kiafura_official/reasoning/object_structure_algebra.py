"""Typed object decomposition and topology operators derived from A-only evidence."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
Point=tuple[int,int]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)
def background_of(grid:Grid)->int:return Counter(value for row in grid for value in row).most_common(1)[0][0]
def components(grid:Grid,color:int,connectivity:int=4)->list[set[Point]]:
    pending={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color};result=[];steps=((-1,0),(1,0),(0,-1),(0,1)) if connectivity==4 else tuple((dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc)
    while pending:
        seed=pending.pop();item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for dr,dc in steps:
                neighbor=(r+dr,c+dc)
                if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
        result.append(item)
    return result

def factor_polyomino(grid:Grid,source_color:int,square_color:int,line_color:int)->Grid:
    background=background_of(grid);cells={(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==source_color}
    if not cells:raise ValueError('polyomino absent')
    placements=[]
    for row,column in sorted(cells):
        candidates=((frozenset(((row,column),(row,column+1),(row+1,column),(row+1,column+1))),square_color),(frozenset(((row,column),(row,column+1),(row,column+2))),line_color),(frozenset(((row,column),(row+1,column),(row+2,column))),line_color))
        for placement in candidates:
            if set(placement[0])<=cells and placement not in placements:placements.append(placement)
    by_cell={cell:[] for cell in cells}
    for placement in placements:
        for cell in placement[0]:by_cell[cell].append(placement)
    def cover(remaining,selected):
        if not remaining:return selected
        cell=min(remaining,key=lambda candidate:sum(set(piece)<=remaining for piece,_ in by_cell[candidate]))
        for piece,color in by_cell[cell]:
            if set(piece)<=remaining:
                answer=cover(remaining-set(piece),selected+[(piece,color)])
                if answer is not None:return answer
        return None
    factorization=cover(cells,[])
    if factorization is None:raise ValueError('factorization absent')
    output=[[background]*len(grid[0]) for _ in grid]
    for placement,color in factorization:
        for r,c in placement:output[r][c]=color
    return as_grid(output)

def normalize_rectangles(grid:Grid,output_height:int)->Grid:
    background=background_of(grid);height,width=len(grid),len(grid[0]);pending={(r,c) for r in range(height) for c in range(width) if grid[r][c]!=background};rectangles=[]
    while pending:
        seed=pending.pop();item={seed};stack=[seed]
        while stack:
            r,c=stack.pop()
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in pending:pending.remove(neighbor);item.add(neighbor);stack.append(neighbor)
        rows=[r for r,_ in item];columns=[c for _,c in item];top,bottom,left,right=min(rows),max(rows),min(columns),max(columns)
        if len(item)!=(bottom-top+1)*(right-left+1):raise ValueError('non-rectangle component')
        rectangles.append((left,bottom-top+1,right-left+1,grid[seed[0]][seed[1]]))
    if not rectangles:raise ValueError('rectangles absent')
    rectangles.sort();top_width=sum(max(h,w) for _,h,w,_ in rectangles);bottom_width=sum(min(h,w) for _,h,w,_ in rectangles);canvas_width=max(top_width,bottom_width)+len(rectangles)-1;output=[[background]*canvas_width for _ in range(output_height)];column=0
    for _,rect_height,rect_width,color in rectangles:
        landscape_height,landscape_width=min(rect_height,rect_width),max(rect_height,rect_width)
        for r in range(landscape_height):
            for offset in range(landscape_width):output[r][column+offset]=color
        column+=landscape_width+1
    column=0
    for _,rect_height,rect_width,color in rectangles:
        portrait_height,portrait_width=max(rect_height,rect_width),min(rect_height,rect_width)
        if portrait_height>output_height:raise ValueError('canvas too short')
        for r in range(output_height-portrait_height,output_height):
            for offset in range(portrait_width):output[r][column+offset]=color
        column+=portrait_width+1
    return as_grid(output)

def glyph_signature(component:set[Point])->tuple[bool,bool,bool,bool]:
    top,left=min(r for r,_ in component),min(c for _,c in component);local={(r-top,c-left) for r,c in component};height=max(r for r,_ in local)+1;width=max(c for _,c in local)+1;outside={(-1,-1)};flood=[(-1,-1)]
    while flood:
        row,column=flood.pop()
        for neighbor in ((row-1,column),(row+1,column),(row,column-1),(row,column+1)):
            if -1<=neighbor[0]<=height and -1<=neighbor[1]<=width and neighbor not in outside and neighbor not in local:outside.add(neighbor);flood.append(neighbor)
    holes=any((r,c) not in local and (r,c) not in outside for r in range(height) for c in range(width));edges=sum((r+1,c) in local for r,c in local)+sum((r,c+1) in local for r,c in local);is_one=any(all((r,c) in local for c in range(width)) and len({c for other,c in local if other!=r})<=1 for r in range(height)) or any(all((r,c) in local for r in range(height)) and len({r for r,other in local if other!=c})<=1 for c in range(width));degrees={point:sum((point[0]+dr,point[1]+dc) in local for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))) for point in local};ends=[point for point,degree in degrees.items() if degree==1];close_ends=len(ends)==2 and abs(ends[0][0]-ends[1][0])+abs(ends[0][1]-ends[1][1])<=max(height,width)+2;return holes,edges>=len(local),is_one,close_ends
def classify_glyphs(grid:Grid,source_color:int,mapping:tuple[tuple[tuple[bool,bool,bool,bool],int],...])->Grid:
    lookup=dict(mapping);output=[list(row) for row in grid];items=components(grid,source_color,4)
    if not items:raise ValueError('glyphs absent')
    for item in items:
        signature=glyph_signature(item)
        if signature not in lookup:raise ValueError('unseen topology')
        for r,c in item:output[r][c]=lookup[signature]
    return as_grid(output)
def glyph_mapping(demos,source_color):
    mapping={}
    for source,expected in demos:
        for item in components(source,source_color,4):
            colors={expected[r][c] for r,c in item};signature=glyph_signature(item)
            if len(colors)!=1:return None
            color=next(iter(colors))
            if signature in mapping and mapping[signature]!=color:return None
            mapping[signature]=color
    return tuple(sorted(mapping.items(),key=repr)) if mapping else None

def swap_glyph_relation_roles(grid:Grid)->Grid:
    height,width=len(grid),len(grid[0]);background=background_of(grid);colors=sorted({value for row in grid for value in row if value!=background});by_color={color:components(grid,color,8) for color in colors}
    glyph_candidates=[color for color in colors if len(by_color[color])>=2]
    if len(glyph_candidates)!=1 or len(colors)!=2:raise ValueError('roles ambiguous')
    glyph_color=glyph_candidates[0];path_color=next(color for color in colors if color!=glyph_color);glyphs=by_color[glyph_color]
    if len(by_color[path_color])!=1:raise ValueError('path fragmented')
    path=by_color[path_color][0];prototype=glyphs[0];prototype_center=((min(r for r,_ in prototype)+max(r for r,_ in prototype))//2,(min(c for _,c in prototype)+max(c for _,c in prototype))//2);offsets=[(r-prototype_center[0],c-prototype_center[1]) for r,c in prototype];endpoints=[]
    for r,c in path:
        degree=sum((r+dr,c+dc) in path for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc)
        if degree==1:endpoints.append((r,c))
    centers=[((min(r for r,_ in glyph)+max(r for r,_ in glyph))//2,(min(c for _,c in glyph)+max(c for _,c in glyph))//2) for glyph in glyphs];output=[[background]*width for _ in range(height)]
    for index,(first_row,first_column) in enumerate(centers):
        for second_row,second_column in centers[index+1:]:
            dr,dc=second_row-first_row,second_column-first_column;row_step=0 if dr==0 else (1 if dr>0 else -1);column_step=0 if dc==0 else (1 if dc>0 else -1);steps=max(abs(dr),abs(dc))
            for step in range(steps+1):output[first_row+step*row_step][first_column+step*column_step]=path_color
    for center_row,center_column in endpoints:
        for dr,dc in offsets:
            r,c=center_row+dr,center_column+dc
            if not (0<=r<height and 0<=c<width):raise ValueError('glyph outside canvas')
            output[r][c]=glyph_color
    return as_grid(output)

@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        values=dict(self.bindings)
        if self.graph=='factor_polyomino':return factor_polyomino(grid,values['source_color'],values['square_color'],values['line_color'])
        if self.graph=='normalize_rectangles':return normalize_rectangles(grid,values['output_height'])
        if self.graph=='classify_glyphs':return classify_glyphs(grid,values['source_color'],values['mapping'])
        if self.graph=='swap_glyph_relation_roles':return swap_glyph_relation_roles(grid)
        raise ValueError('unknown graph')
def candidates(demos):
    same=all(len(a)==len(b) and len(a[0])==len(b[0]) for a,b in demos);result=set()
    if not same:
        heights={len(expected) for _,expected in demos}
        if len(heights)==1:result.add(Candidate('normalize_rectangles',(('output_height',next(iter(heights))),)))
        return tuple(result)
    common=None;output_palette=set()
    for source,expected in demos:
        background=background_of(source);roles={value for row in source for value in row if value!=background};common=roles if common is None else common&roles;output_palette.update(value for row in expected for value in row if value!=background)
    for source_color in sorted(common or ()):
        mapping=glyph_mapping(demos,source_color)
        if mapping is not None:result.add(Candidate('classify_glyphs',(('mapping',mapping),('source_color',source_color))))
        for square_color,line_color in permutations(sorted(output_palette),2):result.add(Candidate('factor_polyomino',(('line_color',line_color),('source_color',source_color),('square_color',square_color))))
    result.add(Candidate('swap_glyph_relation_roles',()))
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
