import pandas as pd

if __name__ == "__main__":
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

    # Combined
    xls = pd.ExcelFile('./data/raw/dili_dataset.xlsx')
    combined_train_data = pd.read_excel(xls, 'supplementary table S2.1', header=1)
    combined_test_data = pd.read_excel(xls, 'supplementary table S2.2', header=1)

    combined_train = pd.DataFrame()
    combined_train['cid'] = combined_train_data['CID']
    combined_train['smiles'] = combined_train_data['Canonical SMILES']
    combined_train['description'] = combined_train_data['Annotation']
    combined_train['label'] = combined_train_data['Label']

    combined_test = pd.DataFrame()
    combined_test['cid'] = combined_test_data['CID']
    combined_test['smiles'] = combined_test_data['Canonical SMILES']
    combined_test['description'] = combined_test_data['Annotation']
    combined_test['label'] = combined_test_data['Label']

    combined_train.to_csv('./data/raw/combined_train.csv')
    combined_test.to_csv('./data/raw/combined_test.csv')
