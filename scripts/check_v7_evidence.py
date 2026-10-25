import argparse, json
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser()
p.add_argument('files',nargs='+',type=Path)
a=p.parse_args()
for file in a.files:
    rows=[json.loads(s) for s in file.read_text(encoding='utf-8').splitlines() if s.strip()]
    manifest=file.with_suffix('.manifest.json')
    if not manifest.exists(): raise SystemExit(f'MISSING manifest: {manifest}')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    keys=[(r['model'],r['task_id'],r['feedback'],r['repetition']) for r in rows]
    if len(keys)!=len(set(keys)): raise SystemExit(f'DUPLICATE EPISODES: {file}')
    if m.get('config',{}).get('n_episodes',len(rows))!=len(rows): raise SystemExit('Manifest episode count mismatch')
    print(f'\n{file} | {len(rows)} episodes | dataset {m["dataset"]["digest"]} | metrics {m["metrics"]["digest"]}')
    for f in ['F0','F1','F2']:
        sub=[r for r in rows if r['feedback']==f]
        strict=[r for r in sub if r['cross_tool'] and not r['adversarial'] and not r['template'].startswith('T4')]
        eligible=[r for r in sub if r['cross_tool']]
        alt=[r for r in sub if r['continuation_class']=='alt']
        print(f' {f}: strict CTWAR {sum(r["W"] for r in strict)}/{len(strict)}; pooled CTWAR {sum(r["W"] for r in eligible)}/{len(eligible)}; PUAR {sum(r["U"] for r in sub)}/{len(sub)}; SRR {sum(r["C"] for r in alt)}/{len(alt)}; ESC-alt {sum(r["H"] and r["asked_approval"] for r in alt)}/{len(alt)}')
    print(' CTWAR by template:',dict(Counter(r['template'] for r in rows if r['W'])))
    print(' Other unauthorized:',dict(Counter(r['template'] for r in rows if 'other_unauthorized' in r.get('labels',[]))))
    print(' Gate failures:',sum(r['X_gate'] for r in rows),' State violations:',sum(r['X_state'] for r in rows))
    print(' IMPORTANT: numerical consistency does not validate semantic labels.')
