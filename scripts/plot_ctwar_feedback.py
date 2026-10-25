import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('files',nargs='+',type=Path)
p.add_argument('--out',default='figures/ctwar_strict_v7.png')
a=p.parse_args()
try: import matplotlib.pyplot as plt
except ImportError: raise SystemExit('For plotting only: python -m pip install matplotlib')
fig,ax=plt.subplots(figsize=(6.3,3.5))
for f in a.files:
    rows=[json.loads(s) for s in f.read_text(encoding='utf-8').splitlines() if s.strip()]
    vals=[]
    for condition in ('F0','F1','F2'):
        e=[r for r in rows if r['feedback']==condition and r['cross_tool'] and not r['adversarial'] and not r['template'].startswith('T4')]
        vals.append(100*sum(r['W'] for r in e)/len(e) if e else float('nan'))
    ax.plot(['F0','F1','F2'],vals,marker='o',label=rows[0]['model'].split(':')[1].split('-detour')[0])
ax.set_ylabel('Strict CTWAR (%)');ax.set_xlabel('Denial feedback')
ax.set_ylim(bottom=0);ax.legend();ax.grid(alpha=0.2)
fig.tight_layout();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=200);fig.savefig(out.with_suffix('.pdf'))
print('Saved',out,'and',out.with_suffix('.pdf'))
