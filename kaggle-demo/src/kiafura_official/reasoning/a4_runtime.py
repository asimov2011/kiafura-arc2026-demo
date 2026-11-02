"""Frozen A4-only demo-bound, exact-filtered, fail-closed operational union.

The caller supplies a manifest, demonstrations and query inputs. This module
does not read datasets, certified solutions, identifiers or expected queries.
"""
from . import (a4_basic_algebra, a4_region_algebra, a4_rays_algebra,
               a4_move_algebra, a4_pattern_algebra, a4_template_algebra,
               a4_field_algebra, a4_layout_algebra)
from .a4_core import ERRORS

FAMILIES = {
    'basic': a4_basic_algebra, 'region': a4_region_algebra,
    'rays': a4_rays_algebra, 'move': a4_move_algebra,
    'pattern': a4_pattern_algebra, 'template': a4_template_algebra,
    'field': a4_field_algebra, 'layout': a4_layout_algebra,
}


def candidates(demos, memory):
    result=[]
    if not demos:
        return ()
    for family in memory['families']:
        allowed=set(family['graphs'])
        module=FAMILIES[family['name']]
        result.extend(c for c in module.candidates(demos) if c.graph in allowed)
    return tuple(result)


def consensus(candidates, queries):
    regions={}
    rejected=0
    for candidate in candidates:
        try:
            outputs=tuple(candidate.execute(q) for q in queries)
            regions.setdefault(outputs,[]).append((candidate.graph,candidate.bindings))
        except ERRORS:
            rejected+=1
    result={'outputs':None,'regions':len(regions),'candidates':[
        c for cs in regions.values() for c in cs],'query_rejected_candidates':rejected}
    if len(regions)==1:
        result['outputs']=next(iter(regions))
        result['status']='exact_demo_single_region'
    else:
        result['status']='no_candidate' if not regions else 'conflicting_query_regions'
    return result


def resolve(demos,queries,memory):
    return consensus(candidates(demos,memory),queries)
