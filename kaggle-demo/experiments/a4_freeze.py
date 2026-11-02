"""Deterministic A4 freeze; complement evaluation requires this exact manifest."""
import hashlib
import json
import sys
from a4_source_inspect import ROOT

FREEZE=ROOT/'memoria/a4/a4_freeze.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    freeze=json.loads(FREEZE.read_text())
    manifest=freeze['files']
    assert hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()==freeze['freeze_sha256']
    for path,expected in manifest.items():
        assert sha(ROOT/path)==expected,'frozen file changed: '+path
    assert freeze['source_counts']=={'exact':100,'false':0,'abstain':0}
    return freeze


if __name__=='__main__':
    if '--verify' in sys.argv:
        print(verify()['freeze_sha256'])
    else:
        if FREEZE.exists():
            raise SystemExit('freeze already exists; no overwrite')
        result=json.loads((ROOT/'experiments/a4_union_result.json').read_text())
        audit=json.loads((ROOT/'experiments/a4_anti_copy.json').read_text())
        assert result['denominator']==100 and result['counts'].get('exact')==100 and not result['residuals']
        assert audit['status']=='PASS'
        paths=list((ROOT/'src/kiafura_official/reasoning').glob('a4_*.py'))
        paths += [ROOT/p for p in (
            'splits/ct-2-independent-a4-v1.json','memoria/a4/a4_operational_memory.json',
            'experiments/a4_eval.py','experiments/a4_source_inspect.py','experiments/a4_memory.py',
            'experiments/a4_verify.py','experiments/a4_freeze.py','experiments/a4_transfer.py',
            'experiments/a4_union_result.json','experiments/a4_source_inventory.json','experiments/a4_anti_copy.json',
            'experiments/a4_replay_result.json')]
        manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
        freeze={'format':'a4-freeze-v1','source_counts':{'exact':100,'false':0,'abstain':0},
                'files':manifest,'freeze_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'transfer_contract':{'denominator':900,'workers':4,'timeout_seconds':120,
                                     'runtime_changes_after_freeze':False}}
        FREEZE.write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
        print(verify()['freeze_sha256'])
