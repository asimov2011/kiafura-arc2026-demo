"""A5 deterministic freeze: required before any complement grid can be opened."""
import hashlib
import json
import sys
from a5_source_inspect import ROOT

FREEZE=ROOT/'memoria/a5/a5_freeze.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    frozen=json.loads(FREEZE.read_text());manifest=frozen['files']
    assert hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()==frozen['freeze_sha256']
    for path,expected in manifest.items():assert sha(ROOT/path)==expected,'frozen file changed: '+path
    assert frozen['source_counts']=={'exact':100,'false':0,'abstain':0}
    return frozen


if __name__=='__main__':
    if '--verify' in sys.argv:
        print(verify()['freeze_sha256'])
    else:
        assert not FREEZE.exists(),'freeze exists; no overwrite'
        result=json.loads((ROOT/'experiments/a5_union_result.json').read_text())
        audit=json.loads((ROOT/'experiments/a5_anti_copy.json').read_text())
        replay=json.loads((ROOT/'experiments/a5_replay_result.json').read_text())
        assert result['denominator']==100 and result['counts']=={'exact':100,'false':0,'abstain':0} and not result['residuals']
        assert audit['status']==replay['status']=='PASS'
        assert audit['source_results_sha256']==replay['source_results_sha256']
        for p,h in replay['runtime_hashes'].items():assert sha(ROOT/p)==h,p
        paths=list((ROOT/'src/kiafura_official/reasoning').glob('a5_*.py'))
        paths+=list((ROOT/'experiments').glob('a5_*.py'))
        paths += [ROOT/p for p in ['splits/ct-2-independent-a5-v1.json','data/mastery/pool34.py','data/mastery/arc2-mastery-v4-pool417.jsonl',
                                  'memoria/a5/a5_operational_memory.json','experiments/a5_union_result.json','experiments/a5_anti_copy.json',
                                  'experiments/a5_replay_result.json','experiments/a5_source_inventory.json','experiments/a5_baseline_result.json']]
        manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
        frozen={'format':'a5-freeze-v1','source_counts':result['counts'],'files':manifest,
                'freeze_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'transfer_contract':{'denominator':900,'workers':4,'timeout_seconds':120,'runtime_changes_after_freeze':False}}
        FREEZE.write_text(json.dumps(frozen,indent=2,sort_keys=True)+'\n')
        print(verify()['freeze_sha256'])
