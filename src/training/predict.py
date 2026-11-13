"""Score a transaction CSV with a saved baseline bundle."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def predict(bundle, frame):
    x=frame[bundle['features']].copy()
    cats=bundle['categorical_features']
    for c in cats:
        # CSV inference can change a code column from float to int on a small
        # batch. Preserve the spelling used at fit time (e.g. 150.0 vs 150).
        dtype=bundle.get('numeric_category_dtypes',{}).get(c)
        if dtype:
            x[c]=pd.to_numeric(x[c],errors='raise').astype('float64' if dtype=='float' else 'Int64')
        x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
        if bundle['category_levels'] is not None:
            x[c]=x[c].astype(pd.CategoricalDtype(bundle['category_levels'][c]))
    nums=[c for c in x.columns if c not in cats]
    x[nums]=x[nums].astype(np.float32)
    return bundle['model'].predict_proba(x)[:,1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.resolve() in {args.data.resolve(),args.model.resolve()}:
        p.error('Output must not overwrite the input or model')
    bundle=joblib.load(args.model)
    frame=pd.read_csv(args.data,low_memory=False)
    result=pd.DataFrame({'TransactionID':frame.TransactionID,'isFraud':predict(bundle,frame)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Saved {len(result):,} predictions to {args.output}')

if __name__=='__main__':main()
