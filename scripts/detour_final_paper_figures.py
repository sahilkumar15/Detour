#!/usr/bin/env python3
"""Figures from audited, frozen Detour aggregates. No model calls."""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path.cwd()
data=root/'results/trust27_evidence'
out=root/'figures/detour_verified'
out.mkdir(parents=True,exist_ok=True)

def rows(path):
    if not path.exists(): raise SystemExit(f'MISSING: {path}. Run evidence audit first.')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

main=rows(data/'main.csv')
keys={(r['model'],r['feedback']):r for r in main}
models=sorted({r['model'] for r in main},key=lambda s:(0 if '7B' in s else 1 if '14B' in s else 2 if 'Mistral' in s else 3,s))
arms=['F0','F1','F2']
if len(models)!=4 or len(keys)!=12 or any((m,f) not in keys for m in models for f in arms):
    raise SystemExit('STOP: expected 4 models x 3 feedback arms; duplicate or missing result cells.')

def val(m,f,n,d,expected):
    r=keys[(m,f)]
    num=int(r[n]); den=int(r[d])
    if den!=expected or not 0<=num<=den:
        raise SystemExit(f'STOP: invalid {n}/{d} for {m} {f}: {num}/{den}')
    return 100*num/den

fig,ax=plt.subplots(1,3,figsize=(13.2,3.7))
metrics=[('CTWAR_num','CTWAR_den',36,'Strict cross-tool attempts'),
         ('SRR_num','SRR_den',54,'Clean authorized recovery'),
         ('ESC_alt_num','ESC_alt_den',54,'Unnecessary escalation')]
for j,(num,den,n,title) in enumerate(metrics):
    for model in models:
        ax[j].plot(
            arms,
            [val(model,f,num,den,n) for f in arms],
            marker='o',
            linewidth=1.8,
            label=model
        )
    ax[j].set_title(title)
    ax[j].set_ylabel('% of eligible episodes')
    ax[j].set_ylim(0,100)
    ax[j].grid(alpha=.2)

handles,labels=ax[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.05),fontsize=9)
fig.tight_layout(rect=(0,.09,1,1))
fig.savefig(out/'feedback_security_utility.pdf',bbox_inches='tight')
fig.savefig(out/'feedback_security_utility.png',dpi=200,bbox_inches='tight')
plt.close(fig)

baseline=rows(data/'naive_baseline.csv')
if len(baseline)!=12:
    raise SystemExit('STOP: baseline expected 12 rows')

by=defaultdict(lambda: {'oracle':0,'naive':0,'naive_only':0})

for r in baseline:
    m=r['model']
    if (m,r['feedback']) not in keys:
        raise SystemExit('STOP: baseline model/feedback mismatch')
    if int(r['eligible'])!=36:
        raise SystemExit('STOP: baseline eligibility mismatch')

    by[m]['oracle']+=int(r['CTWAR'])
    by[m]['naive']+=int(r['naive'])
    by[m]['naive_only']+=int(r['naive_only'])

if sum(x['oracle'] for x in by.values())!=34 or sum(x['naive'] for x in by.values())!=41:
    raise SystemExit('STOP: evidence differs from frozen 34/41 baseline audit')

fig,ax=plt.subplots(figsize=(8,3.5))
xs=list(range(4))

ax.bar(
    [x-.18 for x in xs],
    [by[m]['oracle'] for m in models],
    .35,
    label='Strict Detour oracle'
)
ax.bar(
    [x+.18 for x in xs],
    [by[m]['naive'] for m in models],
    .35,
    label='Naive changed-tool baseline'
)
ax.set_xticks(xs,models)
ax.set_ylabel('Flagged episodes (F0-F2 pooled)')
ax.set_title('Classifier disagreement: 34 strict vs 41 naive flags')
ax.legend()
fig.tight_layout()
fig.savefig(out/'baseline_disagreement.pdf')
plt.close(fig)

# Optional eligible-template heatmap
tp=data/'templates.csv'

if tp.is_file():
    templates=rows(tp)
    sums=defaultdict(lambda:[0,0])

    for r in templates:
        t=r['template'].split('-',1)[0]
        sums[(r['model'],t)][0]+=int(r['strict_W'])
        sums[(r['model'],t)][1]+=int(r['strict_n'])

    eligible=sorted({
        t for m,t in sums
        if sums[(m,t)][1]>0
    })

    if eligible:
        heat=[]

        for m in models:
            heat.append([
                100*sums[(m,t)][0]/sums[(m,t)][1]
                if sums[(m,t)][1]
                else float('nan')
                for t in eligible
            ])

        fig,ax=plt.subplots(figsize=(7,3.6))
        im=ax.imshow(heat,aspect='auto',vmin=0,vmax=100)

        ax.set_xticks(range(len(eligible)),eligible)
        ax.set_yticks(range(len(models)),models)
        ax.set_title('Strict CTWAR by template (F0-F2 pooled)')

        for y in range(len(models)):
            for x in range(len(eligible)):
                ax.text(
                    x,y,f'{heat[y][x]:.0f}%',
                    ha='center',va='center',fontsize=9
                )

        fig.colorbar(im,ax=ax,label='% of eligible episodes')
        fig.tight_layout()
        fig.savefig(out/'strict_template_heatmap.pdf')
        plt.close(fig)

(out/'FIGURE_README.txt').write_text(
    'Figures regenerated from frozen evidence tables.\n'
    'Naive-only flags are not proven false positives until human adjudication.\n'
    'Strict analysis includes only four independent template mechanisms.\n',
    encoding='utf-8'
)

print('PASS: Verified 12 main result cells; 34 strict vs 41 naive flags')
print('Figure folder:',out)

for p in sorted(out.iterdir()):
    print(' ',p.name)
