"""Source-only inspection of certified A4 programs; never imported by runtime."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_ids():
    return json.loads((ROOT / 'splits/ct-2-independent-a4-v1.json').read_text())['source_ids']


def records():
    allowed = set(source_ids())
    with (ROOT / 'data/mastery/arc2-mastery-v4-pool417.jsonl').open() as stream:
        return {r['task_id']: r for r in map(json.loads, stream) if r['task_id'] in allowed}


def listing(tokens):
    """Pretty-print only: deliberately has no execution or compilation path."""
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
        if token.startswith('NAME:'):
            return token[5:]
        return token

    lines = []
    for token in it:
        if token in ('LET', 'OUT'):
            lines.append(('O' if token == 'OUT' else 'v' + str(len(lines) + 1)) + ' = ' + expr(next(it)))
    return '\n'.join(lines)


def task(tid):
    if tid not in source_ids():
        raise ValueError('A4 source-only access')
    return json.loads((ROOT / 'data/arc-agi-2-training' / (tid + '.json')).read_text())


if __name__ == '__main__':
    rows = records()
    if sys.argv[1] == 'view':
        for tid in sys.argv[2:]:
            print(tid, listing(rows[tid]['tokens']))
            for i, pair in enumerate(task(tid)['train']):
                print('DEMO', i)
                for key in ('input', 'output'):
                    print(key)
                    print('\n'.join(''.join(str(c) if c else '.' for c in row) for row in pair[key]))
    else:
        start, end = map(int, sys.argv[1:3])
        for i, tid in enumerate(source_ids()[start:end], start):
            print('\n', i, tid)
            print(listing(rows[tid]['tokens']))
            print('demos', [(len(p['input']), len(p['input'][0]), len(p['output']), len(p['output'][0])) for p in task(tid)['train']])
