import argparse, csv, json, random
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('files',nargs='+',type=Path)
p.add_argument('--out',default='annotations/v7')
a=p.parse_args()
review=[]; answer=[]
for f in a.files:
    for line in f.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        for s in r.get('steps',[]):
            if s.get('allowed') is not False: continue
            idx=len(review)+1
            cid=f'CTW-{idx:05d}'
            review.append({'case_id':cid,'denied_tool':r.get('denied_tool'),'denied_effects':json.dumps(r.get('denied_effects')),'proposed_tool':s.get('tool'),'proposed_args':json.dumps(s.get('args'),ensure_ascii=False),'proposed_effects':json.dumps(s.get('effects')),'human_same_effect':'','human_still_forbidden':'','human_label':'','notes':''})
            answer.append({'case_id':cid,'source':str(f),'model':r.get('model'),'feedback':r.get('feedback'),'task':r.get('task_id'),'episode_W':r.get('W'),'automatic_step_label':s.get('label')})
random.Random(2027).shuffle(review)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for name,rows in [('review_blinded.csv',review),('answer_key_private.csv',answer)]:
    dest=out/name
    with dest.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else ['case_id']); w.writeheader();w.writerows(rows)
    print(dest, len(rows),'cases')
print('Keep answer_key_private.csv away from annotators. Add independent labels; this script does NOT validate equivalence.')
