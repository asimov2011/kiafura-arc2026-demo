"""Halfway milestone: frozen A100 + A2 + A300 + A4 + A5 on public B1000.

Each member retains its original resolver and memory. A member abstention is
not an emission; agreeing emissions produce one answer, disagreement abstains.
No source membership routing, score oracle, memory changes or optimization.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a2_baseline import build_memories as load_a100, FAMILY_FILES
from a2_union_eval import build_memories as load_a2, A2_FAMILY_FILES
from a300_common import load_memories as load_a300
from a4_freeze import verify as verify_a4
from a5_freeze import verify as verify_a5
from kiafura_official.reasoning.operational_dsd import resolve as legacy_resolve
from kiafura_official.reasoning.a4_runtime import resolve as resolve_a4
from kiafura_official.reasoning.a5_runtime import resolve as resolve_a5

RUNS=('A100','A2','A300','A4','A5')
MEMORIES={}
WORKERS=4
MEMBER_TIMEOUT=120
FAMILY_BUDGET=2_000_000
MANIFEST=ROOT/'experiments/halfway_b1000_manifest.json'
PROGRESS=ROOT/'experiments/halfway_b1000_progress.jsonl'
RESULT=ROOT/'experiments/halfway_b1000_result.json'


def grid(g):return tuple(tuple(row) for row in g)


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_inputs():
    verify_a4();verify_a5()
    m=json.loads((ROOT/'memoria/a300/manifest.json').read_text())
    for path,h in m['runtime_sha256'].items():assert sha(ROOT/path)==h,path
    for name,h in m['memory_file_sha256'].items():assert sha(ROOT/'memoria/a300'/name)==h,name
    return m


def initialize():
    global MEMORIES
    MEMORIES={'A100':load_a100(),'A2':load_a2(),'A300':load_a300(),
              'A4':json.loads((ROOT/'memoria/a4/a4_operational_memory.json').read_text()),
              'A5':json.loads((ROOT/'memoria/a5/a5_operational_memory.json').read_text())}


def consensus(members):
    regions={}
    for name,member in members.items():
        if member['outputs'] is not None:
            outputs=tuple(grid(q) for q in member['outputs'])
            regions.setdefault(outputs,[]).append(name)
    outputs=next(iter(regions)) if len(regions)==1 else None
    return {'outputs':outputs,'regions':len(regions),
            'emitting_runs':sorted(n for ns in regions.values() for n in ns),
            'status':'verified' if outputs is not None else 'conflict' if regions else 'no_emission'}


def resolve_union(demos,queries):
    members={}
    def timeout(*_):raise TimeoutError('frozen member exceeded 120 seconds')
    signal.signal(signal.SIGALRM,timeout)
    for name in RUNS:
        signal.alarm(MEMBER_TIMEOUT)
        try:
            if name in ('A100','A2','A300'):
                r=legacy_resolve(demos,queries,MEMORIES[name],FAMILY_BUDGET)
            else:
                r=(resolve_a4 if name=='A4' else resolve_a5)(demos,queries,MEMORIES[name])
            members[name]={'outputs':r.get('outputs'),'status':r['status']}
        except TimeoutError:
            members[name]={'outputs':None,'status':'timeout'}
        except Exception as exc:
            members[name]={'outputs':None,'status':'error','error':type(exc).__name__+': '+str(exc)}
        finally:signal.alarm(0)
    return {**consensus(members),'members':members}


def run(tid):
    t=json.loads((ROOT/'data/arc-agi-2-training'/(tid+'.json')).read_text())
    demos=[(grid(p['input']),grid(p['output'])) for p in t['train']]
    queries=[grid(p['input']) for p in t['test']]
    r=resolve_union(demos,queries)
    expected=tuple(grid(p['output']) for p in t['test'])
    verdict='abstain' if r['outputs'] is None else 'exact' if r['outputs']==expected else 'false'
    return {'task_id':tid,'verdict':verdict,**r}


def negative_test():
    one={'outputs':(((1,),),)};two={'outputs':(((2,),),)};none={'outputs':None}
    assert consensus({'A':one,'B':one,'C':none})['outputs']==(((1,),),)
    assert consensus({'A':one,'B':two})['outputs'] is None
    assert consensus({'A':none})['outputs'] is None
    assert consensus({'A':one,'B':{'outputs':None,'status':'timeout'}})['outputs']==(((1,),),)


if __name__=='__main__':
    negative_test()
    if '--test' in sys.argv:
        print('PASS: agreement, conflict, empty and non-emitting timeout semantics')
        raise SystemExit(0)
    assert not any(p.exists() for p in (MANIFEST,PROGRESS,RESULT)), 'existing milestone evidence; no overwrite'
    a300=verify_inputs();initialize()
    ids=sorted(p.stem for p in (ROOT/'data/arc-agi-2-training').glob('*.json'))
    assert len(ids)==1000
    memory_hashes={n:digest(m) for n,m in MEMORIES.items()}
    # Snapshot all reasoning implementation files without importing extra capabilities.
    paths=list((ROOT/'src/kiafura_official/reasoning').glob('*.py'))
    paths+=[ROOT/'src/kiafura_official/arc_perception.py',Path(__file__),
            ROOT/'experiments/a2_baseline.py',ROOT/'experiments/a2_union_eval.py',ROOT/'experiments/a300_common.py']
    paths+=[ROOT/'memoria'/p for p in ['ct-2-operational-memory-v1.json','ct6-causal-ssa-memory-v2.json',*FAMILY_FILES.values(),
            'a2/ct-2-a2-operational-memory-v1.json','a2/ct-2-a2-causal-ssa-memory-v1.json',*A2_FAMILY_FILES.values(),
            'a4/a4_operational_memory.json','a5/a5_operational_memory.json']]
    paths+=[ROOT/'memoria/a300'/p for p in a300['memory_files'].values()]
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    manifest={'format':'halfway-b1000-v1','runs':RUNS,'denominator':1000,'task_ids':ids,'task_ids_sha256':digest(ids),
              'implementation_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'member_memory_sha256':memory_hashes,'files':files,'workers':WORKERS,'per_member_timeout_seconds':MEMBER_TIMEOUT,
              'legacy_family_work_budget':FAMILY_BUDGET,'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
              'union_policy':'consensus of emitted predictions from unchanged member resolvers; no emission for abstention/timeout/error',
              'negative_test':'PASS','no_source_membership_routing':True,'no_new_optimization':True}
    manifest['manifest_sha256']=digest(manifest)
    MANIFEST.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    rows={};counts=Counter();start=time.monotonic()
    print('START B1000',manifest['manifest_sha256'],flush=True)
    with PROGRESS.open('x') as log,ProcessPoolExecutor(max_workers=WORKERS,initializer=initialize) as executor:
        pending={executor.submit(run,tid):tid for tid in ids}
        for future in as_completed(pending):
            r=future.result();rows[r['task_id']]=r;counts[r['verdict']]+=1
            log.write(json.dumps(r,sort_keys=True)+'\n');log.flush()
            if len(rows)%50==0:print(len(rows),dict(counts),flush=True)
    verify_inputs()
    for p,h in files.items():assert sha(ROOT/p)==h,'input modified during measurement: '+p
    initialize();assert {n:digest(m) for n,m in MEMORIES.items()}==memory_hashes
    result={'denominator':1000,'runs':RUNS,'counts':{k:counts[k] for k in ('exact','false','abstain')},
            'manifest_sha256':manifest['manifest_sha256'],'elapsed_seconds':round(time.monotonic()-start,3),
            'status_counts':dict(Counter(r['status'] for r in rows.values())),
            'member_timeout_counts':{n:sum(r['members'][n]['status']=='timeout' for r in rows.values()) for n in RUNS},
            'member_error_counts':{n:sum(r['members'][n]['status']=='error' for r in rows.values()) for n in RUNS},
            'unchanged_input_hashes':'PASS','tasks':[rows[tid] for tid in ids]}
    result['records_sha256']=digest(result['tasks'])
    RESULT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('FINAL B1000',result['counts'],result['status_counts'],flush=True)
