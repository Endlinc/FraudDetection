import unittest
import numpy as np
import pandas as pd
from src.training.train_baselines import prepare_data


class TrainingContractTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'TransactionID':range(10), 'TransactionDT':[1,2,3,4,5,6,7,8,8,9],
                             'isFraud':[0,1]*5,'TransactionAmt':np.arange(10,dtype=float),
                             'ProductCD':['A']*7+['new']*3})

    def test_time_ties_and_feature_exclusion(self):
        train,valid,xt,xv,features,cats,nums=prepare_data(self.frame(),.2)
        self.assertLess(train.TransactionDT.max(), valid.TransactionDT.min())
        self.assertEqual(len(train),7)
        self.assertEqual(features,['TransactionAmt','ProductCD'])
        self.assertFalse({'TransactionID','TransactionDT','isFraud'} & set(xt.columns))
        self.assertEqual(list(xt.columns),list(xv.columns))
        self.assertNotIn('new',xt.ProductCD.unique())
        self.assertIn('new',xv.ProductCD.unique())

    def test_duplicate_keys_rejected(self):
        df=self.frame();df.loc[1,'TransactionID']=0
        with self.assertRaisesRegex(ValueError,'unique'):prepare_data(df,.2)

    def test_invalid_label_rejected(self):
        df=self.frame();df.loc[1,'isFraud']=np.nan
        with self.assertRaisesRegex(ValueError,'only 0 and 1'):prepare_data(df,.2)

if __name__=='__main__':unittest.main()
