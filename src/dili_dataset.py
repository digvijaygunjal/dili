import numpy as np
import pandas as pd

xls = pd.ExcelFile('./data/raw/dili_dataset.xlsx')
ncrt_train_data = pd.read_excel(xls, 'supplementary table S1.1', header=1)
ncrt_test_data = pd.read_excel(xls, 'supplementary table S1.2', header=1)

ncrt_train = pd.DataFrame()
ncrt_train['cid'] = ncrt_train_data['CID']
ncrt_train['smiles'] = ncrt_train_data['Canonical SMILES']
ncrt_train['description'] = ncrt_train_data['FDA-approved drug annotation']
ncrt_train['label'] = ncrt_train_data['Label']

ncrt_test = pd.DataFrame()
ncrt_test['cid'] = ncrt_test_data['CID']
ncrt_test['smiles'] = ncrt_test_data['Canonical SMILES']
ncrt_test['description'] = ncrt_test_data['FDA-approved drug annotation']
ncrt_test['label'] = ncrt_test_data['Label']

ncrt_train.to_csv('./data/raw/ncrt_train.csv')
ncrt_test.to_csv('./data/raw/ncrt_test.csv')
