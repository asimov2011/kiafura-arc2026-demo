"""Typed hierarchical panel and tile operators derived from A-only evidence."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Any,Sequence
Grid=tuple[tuple[int,...],...]
def as_grid(value:Sequence[Sequence[int]])->Grid:return tuple(tuple(int(cell) for cell in row) for row in value)

def decode_local_bindings(grid:Grid,background:int)->Grid:
    nonbackground=[value for row in grid for value in row if value!=background]
    if not nonbackground:raise ValueError('material absent')
    base=Counter(nonbackground).most_common(1)[0][0];active_rows=[r for r,row in enumerate(grid) if any(value!=background for value in row)];active_columns=[c for c in range(len(grid[0])) if any(grid[r][c]!=background for r in range(len(grid)))];top,bottom,left,right=min(active_rows),max(active_rows),min(active_columns),max(active_columns);separator_rows=[r for r in range(top,bottom+1) if all(value==background for value in grid[r])];separator_columns=[c for c in range(left,right+1) if all(grid[r][c]==background for r in range(top,bottom+1))];tile_height=separator_rows[0]-top if separator_rows else bottom-top+1;tile_width=separator_columns[0]-left if separator_columns else right-left+1
    if tile_height<=0 or tile_width<=0:raise ValueError('tile geometry absent')
    bindings={}
    for r,row in enumerate(grid):
        for c,value in enumerate(row):
            if value not in (background,base):bindings[((r-top)%(tile_height+1)//2,(c-left)%(tile_width+1)//2)]=value
    if not bindings:raise ValueError('bindings absent')
    tile_rows=max(r for r,_ in bindings)+1;tile_columns=max(c for _,c in bindings)+1;template=[grid[top+r][left:left+tile_width] for r in range(tile_height)];output_height=tile_rows*tile_height+tile_rows-1;output_width=tile_columns*tile_width+tile_columns-1;output=[[base]*output_width for _ in range(output_height)]
    for tile_row in range(tile_rows):
        for tile_column in range(tile_columns):
            color=bindings.get((tile_row,tile_column),base)
            for r in range(tile_height):
                for c in range(tile_width):output[tile_row*(tile_height+1)+r][tile_column*(tile_width+1)+c]=color if template[r][c]!=background else base
    return as_grid(output)

def propagate_canonical_macro_tile(grid:Grid,separator:int,excluded:int)->Grid:
    height,width=len(grid),len(grid[0]);separator_rows=[r for r in range(height) if sum(value==separator for value in grid[r])>=width//2];separator_columns=[c for c in range(width) if sum(grid[r][c]==separator for r in range(height))>=height//2]
    def ranges(separators,limit):
        result=[];start=0
        for stop in [*separators,limit]:
            if start<stop:result.append((start,stop))
            start=stop+1
        return result
    row_ranges=ranges(separator_rows,height);column_ranges=ranges(separator_columns,width);tiles=[]
    for top,bottom in row_ranges:
        for left,right in column_ranges:
            tile=tuple(tuple(grid[r][c] for c in range(left,right)) for r in range(top,bottom))
            if excluded not in {value for row in tile for value in row}:tiles.append(tile)
    if not tiles:raise ValueError('canonical tile absent')
    shapes={(len(tile),len(tile[0])) for tile in tiles if tile and tile[0]}
    if len(shapes)!=1:raise ValueError('nonuniform tiles')
    canonical=Counter(tiles).most_common(1)[0][0];base=Counter(value for row in canonical for value in row).most_common(1)[0][0];output=[[separator if r in separator_rows or c in separator_columns else base for c in range(width)] for r in range(height)];active={(r,c) for r,row in enumerate(canonical) for c,value in enumerate(row) if value!=base}
    for macro_row,macro_column in active:
        if macro_row>=len(row_ranges) or macro_column>=len(column_ranges):raise ValueError('macro address outside partition')
        top,_=row_ranges[macro_row];left,_=column_ranges[macro_column]
        for r,row in enumerate(canonical):
            for c,value in enumerate(row):output[top+r][left+c]=value
    return as_grid(output)

def arrange_legend_glyphs(grid:Grid,background:int,frame_color:int,output_height:int,output_width:int)->Grid:
    frame=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==frame_color]
    if not frame:raise ValueError('legend frame absent')
    top,bottom=min(r for r,_ in frame),max(r for r,_ in frame);left,right=min(c for _,c in frame),max(c for _,c in frame);markers=[(r,c,grid[r][c]) for r in range(top+1,bottom) for c in range(left+1,right) if grid[r][c] not in (background,frame_color)]
    if not markers:raise ValueError('legend markers absent')
    output=[[background]*output_width for _ in range(output_height)]
    for marker_row,marker_column,color in markers:
        cells=[(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value==color and not (top<=r<=bottom and left<=c<=right)]
        if not cells:raise ValueError('glyph absent')
        glyph_top,glyph_bottom=min(r for r,_ in cells),max(r for r,_ in cells);glyph_left,glyph_right=min(c for _,c in cells),max(c for _,c in cells);glyph=[row[glyph_left:glyph_right+1] for row in grid[glyph_top:glyph_bottom+1]];vertical_split=(top+bottom)//2;horizontal_split=(left+right)//2;origin_row=0 if marker_row<vertical_split else output_height-len(glyph);origin_column=0 if marker_column<horizontal_split else output_width-len(glyph[0])
        for r,row in enumerate(glyph):
            for c,value in enumerate(row):
                if value==color:output[origin_row+r][origin_column+c]=color
    return as_grid(output)

def extrapolate_panel_trajectory(grid:Grid)->Grid:
    height,width=len(grid),len(grid[0]);border=grid[0][0];separators=[c for c in range(width) if all(grid[r][c]==border for r in range(height))];spans=[(left,right) for left,right in zip(separators,separators[1:]) if right>left+1]
    if not spans:raise ValueError('panels absent')
    fill=Counter(grid[r][c] for r in range(1,height-1) for c in range(1,width-1)).most_common(1)[0][0];objects=[[(r,c-left) for r in range(1,height-1) for c in range(left+1,right) if grid[r][c]!=fill] for left,right in spans];useful=[points for points in objects if points]
    if not useful:raise ValueError('trajectory absent')
    anchors=[(min(r for r,_ in points),min(c for _,c in points)) for points in useful];row_step,column_step=(anchors[-1][0]-anchors[-2][0],anchors[-1][1]-anchors[-2][1]) if len(anchors)>1 else (0,0);source=useful[-1];anchor_row,anchor_column=anchors[-1];target_row,target_column=anchor_row+row_step,anchor_column+column_step;source_panel_left=separators[len(useful)-1];object_color=grid[source[0][0]][source_panel_left+source[0][1]]
    if object_color==border and row_step:target_row+=1 if row_step>0 else -1
    panel_width=spans[0][1]-spans[0][0]+1;output=[[border]*panel_width]+[[border]+[fill]*(panel_width-2)+[border] for _ in range(height-2)]+[[border]*panel_width]
    for r,c in source:
        destination_row=target_row+(r-anchor_row);destination_column=target_column+(c-anchor_column)
        if 0<destination_row<height-1 and 0<destination_column<panel_width-1:output[destination_row][destination_column]=object_color
    return as_grid(output)

@dataclass(frozen=True)
class Candidate:
    graph:str;bindings:tuple[tuple[str,Any],...]
    def execute(self,grid):
        values=dict(self.bindings)
        if self.graph=='decode_local_bindings':return decode_local_bindings(grid,values['background'])
        if self.graph=='propagate_canonical_macro_tile':return propagate_canonical_macro_tile(grid,values['separator'],values['excluded'])
        if self.graph=='arrange_legend_glyphs':return arrange_legend_glyphs(grid,values['background'],values['frame_color'],values['output_height'],values['output_width'])
        if self.graph=='extrapolate_panel_trajectory':return extrapolate_panel_trajectory(grid)
        raise ValueError('unknown graph')
def candidates(demos):
    common=None;union=set();shapes={(len(expected),len(expected[0])) for _,expected in demos};result={Candidate('extrapolate_panel_trajectory',())}
    for source,_ in demos:
        palette={value for row in source for value in row};union.update(palette);common=palette if common is None else common&palette
    colors=tuple(sorted(common or ()))
    for background in colors:result.add(Candidate('decode_local_bindings',(('background',background),)))
    for separator in colors:
        for excluded in sorted(union-{separator}):result.add(Candidate('propagate_canonical_macro_tile',(('excluded',excluded),('separator',separator))))
    if len(shapes)==1:
        output_height,output_width=next(iter(shapes))
        for background,frame_color in permutations(colors,2):result.add(Candidate('arrange_legend_glyphs',(('background',background),('frame_color',frame_color),('output_height',output_height),('output_width',output_width))))
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
