"""Run from the repository root: PYTHONPATH=. .venv/bin/python experiments/transaction_baselines_20261113/verify.py"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from src.training.predict import predict

out=Path('experiments/transaction_baselines_20261113')
art=Path('data/model_runs/transaction_baselines_20261113')
c=json.loads((out/'config.json').read_text())
p=pd.read_csv(art/'validation_predictions.csv',float_precision='round_trip')
results=json.loads((out/'metrics.json').read_text())
assert len(results)==3
assert len(p)==118108 and not p.TransactionID.duplicated().any()
assert c['feature_count']==391 and not set(c['features'])&{'TransactionID','TransactionDT','isFraud'}
ids=set(p.TransactionID.iloc[:256]);parts=[]
for chunk in pd.read_csv('data/train_transaction.csv',chunksize=100000,low_memory=False):
 selected=chunk[chunk.TransactionID.isin(ids)]
 if len(selected):parts.append(selected)
 if sum(map(len,parts))==len(ids):break
sample=pd.concat(parts).set_index('TransactionID').loc[p.TransactionID.iloc[:256]].reset_index()
checks=[]
for result in results:
 name=result['model'];bundle=joblib.load(art/f'{name}.joblib')
 assert bundle['numeric_category_dtypes']==c['numeric_category_dtypes']
 assert bundle['features']==c['features']
 np.testing.assert_allclose(predict(bundle,sample),p[name].iloc[:256],atol=1e-12)
 for metric,fn in [('roc_auc',roc_auc_score),('average_precision',average_precision_score),('log_loss',log_loss)]:
  assert abs(fn(p.isFraud,p[name])-result[metric])<1e-12
 # Simulate a batch in which numeric categories have different CSV-inferred dtypes.
 alternate=sample.copy()
 for col in c['numeric_category_dtypes']:
  alternate[col]=alternate[col].astype('Int64')
 np.testing.assert_allclose(predict(bundle,alternate),p[name].iloc[:256],atol=1e-12)
 checks.append({'model':name,'metric_recomputation':'passed','saved_model_prediction_rows':256,'categorical_dtype_invariance':'passed'})
base=c['split']['train_fraud']/c['split']['train_rows']
verification={'checks':checks,'rows':len(p),'source_feature_count':391,'excluded_features_absent':True,
 'constant_training_prevalence_baseline':{'roc_auc':.5,'average_precision':float(p.isFraud.mean()),'log_loss':float(log_loss(p.isFraud,np.full(len(p),base)))},
 'inference_schema_note':'Original numeric categorical dtypes attached after fitting; estimators and predictions unchanged.'}
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(results,indent=2));print('All saved-model, exclusion, dtype and metric checks passed.')
