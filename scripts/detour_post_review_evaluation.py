#!/usr/bin/env python3
"""Four-judge Detour study: private, read-only analysis. Never human labels."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path.cwd()
CASES = ROOT / 'annotations/submission_blinded_v2/cases.csv'
ORACLE = ROOT / 'annotations/private_oracle_v2/machine_labels.csv'
INPUT = ROOT / 'annotations/ai_review_v3'
OUT = ROOT / 'annotations/private_oracle_v2/four_judge_final_audit'
ROLES = 'ABCD'
DIMS = ('operation_match', 'principal_match', 'object_match', 'scope_match')
FIELDS = ('equivalent', 'forbidden')
LABELS = ('yes','no','uncertain')
PILOTS = {'DT-06e647b2c6dbce70', 'DT-13ce2c1b988df604',
          'DT-f6ea6a01f46013e3', 'DT-8b9234b5a040d5c2'}


def load(path):
    if not path.is_file():
        raise ValueError(f'Missing file: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        dr = csv.DictReader(f)
        columns, rows = dr.fieldnames or [], list(dr)
    if 'case_id' not in columns:
        raise ValueError(f'No case_id column: {path}')
    ids = [r['case_id'] for r in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate or empty case ID: {path}')
    return {r['case_id']: r for r in rows}, columns


def lab(v):
    v = str(v).strip().lower()
    return {'true':'yes','false':'no','1':'yes','0':'no','y':'yes','n':'no'}.get(v,v)


def kappa(pairs, categories=LABELS):
    if not pairs: return None
    n = len(pairs)
    observed = sum(a==b for a,b in pairs)/n
    ac = Counter(a for a,b in pairs)
    bc = Counter(b for a,b in pairs)
    expected = sum(ac[c]*bc[c] for c in categories)/n**2
    return (observed-expected)/(1-expected) if expected < 1 else None


def pct(n,d):
    return round(100*n/d,2) if d else None


def save_csv(path, rows, cols):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def consensus(values):
    counts = Counter(values)
    winner, n = counts.most_common(1)[0]
    return (winner if n >= 3 else 'split'), n


def joint(row):
    a, b = row['equivalent'], row['forbidden']
    if a == 'yes' and b == 'yes': return 'yes'
    if 'no' in (a,b): return 'no'
    return 'uncertain'


def main():
    cases, _ = load(CASES)
    oracle, machine_cols = load(ORACLE)
    ids = list(cases)
    if len(ids) != 263 or not PILOTS <= set(ids):
        raise ValueError('Expected the original 263-case queue and four pilot IDs')
    if set(ids) != set(oracle):
        raise ValueError('Case/Oracle ID mismatch')
    if not {'auto_equivalent','auto_forbidden'} <= set(machine_cols):
        raise ValueError('Machine oracle must have auto_equivalent and auto_forbidden')

    reviews = {}
    sources = [CASES, ORACLE]
    for role in ROLES:
        path = INPUT / f'ai_{role.lower()}.csv'
        review, cols = load(path)
        sources.append(path)
        if set(review) != set(ids):
            raise ValueError(f'Reviewer {role}: expected exactly the same 263 case IDs, have {len(review)}')
        needed = {'status', 'equivalent', 'forbidden', *DIMS, 'rationale', 'model_used'}
        if not needed <= set(cols):
            raise ValueError(f'Reviewer {role}: missing {needed-set(cols)}')
        for cid, r in review.items():
            if r['status'] != 'ok':
                raise ValueError(f'Reviewer {role}: unresolved case {cid}')
            if any(lab(r[k]) not in LABELS for k in (*FIELDS,*DIMS)):
                raise ValueError(f'Reviewer {role}: invalid label case {cid}')
            dims = [lab(r[k]) for k in DIMS]
            correct = 'no' if 'no' in dims else ('yes' if all(v=='yes' for v in dims) else 'uncertain')
            if lab(r['equivalent']) != correct:
                raise ValueError(f'Reviewer {role}: inconsistent effect dimensions case {cid}')
            for field in FIELDS:
                r[field] = lab(r[field])
        reviews[role] = review
    for cid, o in oracle.items():
        for field in FIELDS:
            if lab(o[f'auto_{field}']) not in ('yes','no'):
                raise ValueError(f'Machine oracle invalid {cid} {field}')
            o[f'auto_{field}'] = lab(o[f'auto_{field}'])

    study = [cid for cid in ids if cid not in PILOTS]
    assert len(study)==259
    OUT.mkdir(parents=True, exist_ok=True)
    try: OUT.chmod(0o700)
    except OSError: pass

    result = {'provenance': {'study_cases':259, 'total_cases':263, 'pilots_excluded':sorted(PILOTS),
                            'analysis_type':'AI judges compared to frozen machine oracle, NOT human ground truth',
                            'sha256': {str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}},
              'model_distributions':{}, 'machine_comparisons':{}, 'pairs':{}, 'consensus':{}}
    for role in ROLES:
        dist = {}
        for field in FIELDS:
            dist[field] = dict(Counter(reviews[role][cid][field] for cid in study))
        dist['models_used'] = dict(Counter(reviews[role][cid]['model_used'] for cid in study))
        result['model_distributions'][role] = dist

        m = {}
        for field in (*FIELDS,'joint'):
            expected = {(cid): (joint({'equivalent':oracle[cid]['auto_equivalent'],
                                       'forbidden':oracle[cid]['auto_forbidden']}) if field=='joint'
                                    else oracle[cid][f'auto_{field}']) for cid in study}
            predicted = {cid:(joint(reviews[role][cid]) if field=='joint' else reviews[role][cid][field]) for cid in study}
            ct = Counter((expected[cid],predicted[cid]) for cid in study)
            positive = sum(expected[cid]=='yes' for cid in study)
            negative = len(study)-positive
            binary = [(expected[cid],predicted[cid]) for cid in study if predicted[cid]!='uncertain']
            m[field] = {'oracle_yes':positive, 'oracle_no':negative,
                        'AI_yes_on_oracle_yes':ct[('yes','yes')],
                        'AI_no_on_oracle_yes':ct[('yes','no')],
                        'AI_uncertain_on_oracle_yes':ct[('yes','uncertain')],
                        'AI_yes_on_oracle_no':ct[('no','yes')],
                        'AI_no_on_oracle_no':ct[('no','no')],
                        'AI_uncertain_on_oracle_no':ct[('no','uncertain')],
                        'coverage_binary_n':len(binary),
                        'agreement_all_pct': pct(sum(expected[cid]==predicted[cid] for cid in study),len(study)),
                        'agreement_binary_pct':pct(sum(a==b for a,b in binary),len(binary)),
                        'kappa_binary':kappa(binary,('yes','no')),
                        'positive_agreement_all_pct':pct(ct[('yes','yes')],positive)}
        result['machine_comparisons'][role] = m

    for a,b in combinations(ROLES,2):
        pair = {}
        for field in (*FIELDS, *DIMS):
            rows = [(lab(reviews[a][cid][field]),lab(reviews[b][cid][field])) for cid in study]
            binary = [(x,y) for x,y in rows if x!='uncertain' and y!='uncertain']
            pair[field] = {'n':len(rows), 'exact_pct':pct(sum(x==y for x,y in rows),len(rows)),
                           'kappa_3label':kappa(rows), 'binary_n':len(binary),
                           'binary_kappa':kappa(binary,('yes','no'))}
        result['pairs'][a+b] = pair

    triage = []
    for cid in study:
        case = cases[cid]
        row = {'case_id':cid, 'denied_tool':case.get('denied_tool',''),
               'proposed_tool':case.get('proposed_tool',''),
               'task_goal':case.get('task_goal',''),
               'denied_arguments':case.get('denied_arguments',''),
               'proposed_arguments':case.get('proposed_arguments',''),
               'policy_rules':case.get('policy_rules',''),
               'prior_actions':case.get('prior_actions',''),
               'principal_directory':case.get('principal_directory',''),
               'initial_documents':case.get('initial_documents','')}
        flags = []
        for field in (*FIELDS,'joint'):
            labels = [joint(reviews[r][cid]) if field=='joint' else reviews[r][cid][field] for r in ROLES]
            auto = joint({'equivalent':oracle[cid]['auto_equivalent'],
                          'forbidden':oracle[cid]['auto_forbidden']}) if field=='joint' else oracle[cid][f'auto_{field}']
            vote, strength = consensus(labels)
            row[f'oracle_{field}'] = auto
            row[f'consensus_{field}'] = vote
            row[f'consensus_{field}_votes'] = strength
            for role,v in zip(ROLES,labels): row[f'{role}_{field}'] = v
            if vote != 'split' and vote != 'uncertain' and vote != auto and strength==4:
                flags.append('P0_unanimous_vs_oracle_'+field)
            elif vote=='split':
                flags.append('P1_split_'+field)
            elif vote != auto and vote!='uncertain' and strength>=3:
                flags.append('P1_majority_vs_oracle_'+field)
            elif any(v=='uncertain' for v in labels):
                flags.append('P2_uncertain_'+field)
        row['review_priority'] = 'P0' if any(x.startswith('P0_') for x in flags) else ('P1' if any(x.startswith('P1_') for x in flags) else ('P2' if flags else 'P3'))
        row['reasons'] = '|'.join(flags)
        for role in ROLES:
            row[f'{role}_model'] = reviews[role][cid]['model_used']
            row[f'{role}_rationale'] = reviews[role][cid].get('rationale','')
        triage.append(row)
    triage.sort(key=lambda r:(r['review_priority'],r['case_id']))
    result['consensus']['review_priorities'] = dict(Counter(r['review_priority'] for r in triage))
    result['consensus']['unanimous_vs_machine_joint'] = sum('P0_unanimous_vs_oracle_joint' in x['reasons'] for x in triage)
    result['consensus']['human_ground_truth_available'] = False

    save_csv(OUT/'triage_PRIVATE.csv',triage,list(triage[0]))
    (OUT/'four_judge_metrics_PRIVATE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

    lines = ['# Detour AI Judge Validation (NOT human annotation)', '',
             '259 non-pilot action proposals; four previously discussed calibration cases excluded.',
             '', '## Model-label distributions']
    for r in ROLES:
        d = result['model_distributions'][r]
        lines.append(f"- Reviewer {r}: equiv {d['equivalent']}; forbidden {d['forbidden']}; actual models {d['models_used']}")
    lines += ['', '## Agreement with machine oracle (machine is NOT ground truth)',
              '| Judge | Equivalence all-case agreement | Equivalence binary kappa | Oracle-positive equivalence agreement | Forbiddenness all-case agreement | Forbiddenness binary kappa |',
              '|---|---:|---:|---:|---:|---:|']
    for r in ROLES:
        e = result['machine_comparisons'][r]['equivalent']
        f = result['machine_comparisons'][r]['forbidden']
        lines.append(f"| {r} | {e['agreement_all_pct']}% | {e['kappa_binary']} | {e['positive_agreement_all_pct']}% | {f['agreement_all_pct']}% | {f['kappa_binary']} |")
    lines += ['', '## Pairwise AI judge comparison',
              '| Pair | Equivalence exact agreement | Equivalence 3-label kappa | Forbidden exact agreement | Forbidden 3-label kappa |',
              '|---|---:|---:|---:|---:|']
    for pair,met in result['pairs'].items():
        e,f=met['equivalent'],met['forbidden']
        lines.append(f"| {pair} | {e['exact_pct']}% | {e['kappa_3label']} | {f['exact_pct']}% | {f['kappa_3label']} |")
    lines += ['', '## Private discrepancy triage',str(result['consensus']['review_priorities']),
              '', 'Machine-label disagreements are NOT adjudicated model errors.',
              'The 259 selected proposals do not estimate global accuracy on 864 episodes.',
              'A consensus of AI models is not independent ground truth.',
              'Provide real blinded human A/B labels, then independent adjudication.',
              'Do NOT release private reports to blinded human annotators.']
    (OUT/'SUMMARY_PRIVATE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for field in FIELDS:
            fig,ax=plt.subplots(figsize=(7,4))
            xs=list(range(4))
            bottom=[0]*4
            for category in LABELS:
                values=[result['model_distributions'][r][field].get(category,0) for r in ROLES]
                ax.bar(xs,values,bottom=bottom,label=category)
                bottom=[a+b for a,b in zip(bottom,values)]
            ax.set_xticks(xs,[f'{r}' for r in ROLES]);ax.set_ylim(0,259)
            ax.set_ylabel('Case count (259 non-pilot proposals)')
            ax.set_xlabel('AI Reviewer')
            ax.set_title(f'AI judgment distribution: {field}')
            ax.legend();fig.tight_layout()
            fig.savefig(OUT/f'AI_{field}_distribution.pdf')
            plt.close(fig)
    except ImportError:
        print('matplotlib unavailable: CSV and metrics created; charts skipped')

    print('PASS: Four judges complete; 259-case non-pilot comparison')
    print('Private report:', OUT/'SUMMARY_PRIVATE.md')
    print('Private triage:', OUT/'triage_PRIVATE.csv')
    print('\n'.join(lines[:16]))
    print('IMPORTANT: NOT human annotation and NOT independent ground-truth accuracy')


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError) as e:
        print('STOP:',e,file=sys.stderr)
        sys.exit(2)
