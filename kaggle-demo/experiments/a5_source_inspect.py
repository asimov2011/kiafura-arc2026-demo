"""A5 source assets and readable listings. No runtime or lowering path."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / 'splits/ct-2-independent-a5-v1.json'


def source_ids():
    return json.loads(SPLIT.read_text())['source_ids']


def records():
    allowed = set(source_ids())
    return {r['task_id']: r for r in map(json.loads, (ROOT/'data/mastery/arc2-mastery-v4-pool417.jsonl').read_text().splitlines()) if r['task_id'] in allowed}


def programs():
    text = (ROOT/'data/mastery/pool34.py').read_text()
    return {n.name[6:]: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def listing(tokens):
    # Same source-only pretty printer used for A4; no execution/compilation.
    it = iter(tokens)
    def expr(token):
        if token == 'INVOKE':
            fn = expr(next(it))
            n = int(next(it).split(':')[1])
            return fn + '(' + ', '.join(expr(next(it)) for _ in range(n)) + ')'
        if token == 'REF':
            digits = []
            for t in it:
                if t == 'END_REF':
                    return 'v' + ''.join(digits)
                digits.append(t[1:])
        if token.startswith('CALL:'):
            n = int(next(it).split(':')[1])
            return token[5:] + '(' + ', '.join(expr(next(it)) for _ in range(n)) + ')'
        return token[5:] if token.startswith('NAME:') else token
    lines = []
    for token in it:
        if token in ('LET', 'OUT'):
            lines.append(('O' if token == 'OUT' else 'v' + str(len(lines)+1)) + ' = ' + expr(next(it)))
    return '\n'.join(lines)


def task(tid):
    if tid not in source_ids():
        raise ValueError('A5 source-only access')
    return json.loads((ROOT/'data/arc-agi-2-training'/(tid+'.json')).read_text())


def initialize():
    assert not SPLIT.exists(), 'split already frozen'
    previous = json.loads((ROOT/'splits/ct-2-independent-a4-v1.json').read_text())
    pool = list(map(json.loads, (ROOT/'data/mastery/arc2-mastery-v4-pool417.jsonl').read_text().splitlines()))
    old = sorted(r['task_id'] for r in pool if r['task_id'] in previous['remaining_certified_ids'] and r['backend']=='hodel-ssa')
    new = sorted(programs())
    assert len(old)==66 and len(new)==34 and not set(old)&set(new)
    ids = sorted(old+new)
    excluded = set()
    for file,key in [('ct-2-random-100-900-v1.json','a_ids'),('ct-2-random-a2-v1.json','a2_ids'),('ct-2-independent-a300-v1.json','source_ids'),('ct-2-independent-a4-v1.json','source_ids')]:
        excluded.update(json.loads((ROOT/'splits'/file).read_text())[key])
    assert not set(ids)&excluded
    # Filenames only: complement grid contents remain unopened until freeze.
    universe = {p.stem for p in (ROOT/'data/arc-agi-2-training').glob('*.json')}
    assert len(universe)==1000 and set(ids)<=universe
    complement = sorted(universe-set(ids))
    digest = lambda x: hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
    result = {'format':'a5-independent-source-v1','run':'A5','selection':'owner: remaining 66 Hodel + corrected pool34',
              'source_ids':ids,'source_ids_sha256':digest(ids),'complement_ids':complement,'complement_ids_sha256':digest(complement),
              'inputs':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['data/mastery/arc2-mastery-v4-pool417.jsonl','data/mastery/pool34.py']},
              'pool417_source_ids':old,'pool34_source_ids':new,'excluded_runs':['A100','A2','A300','A4'],
              'remaining_certified_ids':['0bb8deee'],'remaining_hodel_count':0}
    SPLIT.write_text(json.dumps(result,indent=2)+'\n')
    print('A5 source 100; complement 900; overlap 0; input split frozen')


if __name__=='__main__':
    if sys.argv[1]=='init':
        initialize()
    else:
        rows, new = records(), programs()
        ids = sys.argv[2:] if sys.argv[1]=='view' else source_ids()[int(sys.argv[1]):int(sys.argv[2])]
        for tid in ids:
            print('\nTASK',tid)
            print(listing(rows[tid]['tokens']) if tid in rows else new[tid])
            t = task(tid)
            print('demos',[(len(p['input']),len(p['input'][0]),len(p['output']),len(p['output'][0])) for p in t['train']])
            if sys.argv[1]=='view':
                for i,p in enumerate(t['train']):
                    print('DEMO',i)
                    for key in ('input','output'):
                        print(key, '\n'+'\n'.join(''.join(str(c) if c else '.' for c in row) for row in p[key]))
