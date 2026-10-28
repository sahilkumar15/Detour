#!/usr/bin/env python3
"""Create a blinded semantic review queue for Detour 'C_plus' artifacts.

Requires human review; no automated semantic accuracy is claimed.
"""
import argparse,csv,json,random
from pathlib import Path

FIELDS=['case_id','legitimate_goal','source_documents','recorded_authorized_writes',
        'semantically_relevant_yes_no_unsure','factually_supported_yes_no_unsure',
        'adequate_completion_yes_no_unsure','rationale']

def write(path,rows,fields):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',nargs='+');p.add_argument('--out',default='annotations/recovery_review');p.add_argument('--max',type=int,default=150)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    try:
        from pda.scenarios import build_suite
        specs={s.task_id:s for s in build_suite()}
    except Exception as ex:raise SystemExit(f'Could not load Detour scenarios: {ex}')
    cases=[]
    for source in a.files:
        for i,line in enumerate(Path(source).read_text(encoding='utf8').splitlines(),1):
            if not line.strip():continue
            r=json.loads(line)
            if r.get('continuation_class')!='alt' or not r.get('C_plus'):continue
            spec=specs[r['task_id']]
            initial=[{'file_name':f.filename,'source_content':f.content} for f in spec.build().cloud_drive.files.values()]
            writes=[{'tool':st['tool'],'args':st.get('args'), 'executed_ok':st.get('executed_ok')}
                for st in r.get('steps',[]) if st.get('allowed') and st.get('executed_ok') and
                st.get('tool') in ('create_file','append_to_file','share_file','send_email','send_direct_message','send_channel_message')]
            cases.append({'source_file':str(source),'source_line':i,'model':r.get('model'),'feedback':r.get('feedback'),'task_id':r['task_id'],
              'W':r.get('W'),'C_plus':r.get('C_plus'),'must_include':bool(r.get('W')),
              'legitimate_goal':spec.goal,'source_documents':json.dumps(initial,ensure_ascii=False),
              'recorded_authorized_writes':json.dumps(writes,ensure_ascii=False)})
    rng=random.Random(2027)
    forced=[r for r in cases if r['must_include']]
    rest=[r for r in cases if not r['must_include']]
    rng.shuffle(rest)
    selected=forced+rest[:max(0,a.max-len(forced))]
    rng.shuffle(selected)
    reviewed=[];key=[]
    for ix,r in enumerate(selected,1):
        ident=f'SR-{ix:05d}'
        rec={f:r.get(f,'') for f in FIELDS};rec['case_id']=ident
        reviewed.append(rec)
        key.append({'case_id':ident,**{k:r[k] for k in ('source_file','source_line','model','feedback','task_id','W','C_plus','must_include')}})
    write(out/'reviewer_recovery.csv',reviewed,FIELDS)
    write(out/'answer_key_private.csv',key,list(key[0]) if key else ['case_id'])
    print('Prepared',len(reviewed),'recovery review cases; includes all',len(forced),'CTWAR+completion cases')
    print('Inspect all source documents and actual authorized write contents before labeling.')
    print('IMPORTANT: end-state may include actions not visible as writes here; check trace/state as needed.')

if __name__=='__main__':main()
