"""Profile transaction-origin columns in the joined CSV; requires pandas and numpy."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def main():
    with (ROOT/'data/train_transaction.csv').open() as f:
        columns=next(csv.reader(f))
    df=pd.read_csv(ROOT/'data/train_joined.csv',usecols=columns,low_memory=False)
    previous=json.loads((ROOT/'analysis/train_profile.json').read_text())
    prior={r['column']:r['missing'] for r in previous['missing']}
    attributes=[]
    for name in columns:
        s=df[name]; clean=s.dropna(); missing=int(s.isna().sum()); unique=int(clean.nunique())
        assert missing==prior[name], name
        family=next((prefix for prefix in ['card','addr','dist','C','D','M','V'] if name.startswith(prefix) and name[len(prefix):].isdigit()),'Core / email')
        categorical=name in ['isFraud','ProductCD','P_emaildomain','R_emaildomain'] or family in ['card','addr','M'] or not pd.api.types.is_numeric_dtype(s)
        row={'name':name,'family':family,'missing':missing,'present':len(clean),'fraction':missing/len(df),'unique':unique,'type':'Categorical / code' if categorical else 'Numeric','role':'Target label' if name=='isFraud' else 'Identifier' if name=='TransactionID' else 'Attribute'}
        if categorical or unique<=20:
            counts=clean.value_counts().sort_values(ascending=False)
            top=counts.iloc[:19] if len(counts)>20 else counts
            row['distribution']=[[str(k),int(n)] for k,n in top.items()]
            if len(counts)>20:row['distribution'].append(['Other values',int(counts.iloc[19:].sum())])
            row['kind']='Value counts' if len(counts)<=20 else 'Top 19 values + Other'
        elif len(clean):
            values=clean.to_numpy(dtype=float)
            assert np.isfinite(values).all(),name
            counts,edges=np.histogram(values,bins=20)
            row['distribution']=[[f'{a:.7g}–{b:.7g}'+(' (incl.)' if i==19 else ''),int(n)] for i,(a,b,n) in enumerate(zip(edges[:-1],edges[1:],counts))]
            row['kind']='20 equal-width bins'
            row['bin_edges']=edges.tolist()
        else:row.update(distribution=[],kind='No observed values')
        if not categorical and len(clean):
            row.update(min=float(clean.min()),median=float(clean.median()),max=float(clean.max()))
        assert sum(n for _,n in row['distribution'])==row['present']
        attributes.append(row)
    out={'rows':len(df),'columns':len(columns),'attributes':attributes,'source':'data/train_joined.csv','column_scope':'Columns present in data/train_transaction.csv','missing_definition':'pandas default NA recognition; counts checked against prior empty/NaN profile','checks':{'all_missing_counts_match_previous_profile':True,'all_distributions_sum_to_present_count':True}}
    (ROOT/'analysis/transaction_attributes.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'rows':len(df),'columns':len(columns),'complete_columns':sum(r['missing']==0 for r in attributes),'over_half_missing':sum(r['fraction']>.5 for r in attributes),'top_missing':sorted([(r['name'],r['missing'],r['fraction']) for r in attributes],key=lambda x:x[1],reverse=True)[:5]}))

if __name__=='__main__':main()
