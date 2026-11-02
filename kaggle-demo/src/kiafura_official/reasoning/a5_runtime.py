"""A5-only exact-demo candidates and fail-closed query consensus.

No dataset access, programs, expected query answers or cross-run capability.
"""
from . import (a5_basic_algebra, a5_panel_algebra, a5_region_algebra,
              a5_rays_algebra, a5_move_algebra, a5_layout_algebra,
              a5_template_algebra, a5_pattern_algebra)
from .a5_core import ERRORS

FAMILIES={'basic':a5_basic_algebra,'panel':a5_panel_algebra,
          'region':a5_region_algebra,'rays':a5_rays_algebra,
          'move':a5_move_algebra,'layout':a5_layout_algebra,
          'template':a5_template_algebra,'pattern':a5_pattern_algebra}


def candidates(demos,memory):
    if not demos:return ()
    result=[]
    for family in memory['families']:
        allowed=set(family['graphs'])
        result.extend(c for c in FAMILIES[family['name']].candidates(demos) if c.graph in allowed)
    return tuple(result)


def consensus(candidates,queries):
    regions={};rejected=0
    for c in candidates:
        try:
            out=tuple(c.execute(q) for q in queries)
            regions.setdefault(out,[]).append((c.graph,c.bindings))
        except ERRORS:rejected+=1
    return {'outputs':next(iter(regions)) if len(regions)==1 else None,
            'regions':len(regions),'candidates':[c for cs in regions.values() for c in cs],
            'query_rejected_candidates':rejected,
            'status':'exact_demo_single_region' if len(regions)==1 else 'no_candidate' if not regions else 'conflicting_query_regions'}


def resolve(demos,queries,memory):
    return consensus(candidates(demos,memory),queries)
