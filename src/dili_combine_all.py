import pandas as pd

if __name__ == "__main__":
    xls = pd.ExcelFile('./data/raw/dili_dataset.xlsx')
    ncrt_train_data = pd.read_excel(xls, 'supplementary table S1.1', header=1)
    ncrt_test_data = pd.read_excel(xls, 'supplementary table S1.2', header=1)

    ncrt_train = pd.DataFrame()
    ncrt_train['smiles'] = ncrt_train_data['Canonical SMILES']
    ncrt_train['label'] = ncrt_train_data['Label']

    ncrt_test = pd.DataFrame()
    ncrt_test['smiles'] = ncrt_test_data['Canonical SMILES']
    ncrt_test['label'] = ncrt_test_data['Label']

    # ncrt_train.to_csv('./data/raw/ncrt_train.csv')
    # ncrt_test.to_csv('./data/raw/ncrt_test.csv')


    liew_train_data = pd.read_excel(xls, 'supplementary table S3.1', header=1)
    liew_test_data = pd.read_excel(xls, 'supplementary table S3.2', header=1)

    liew_train = pd.DataFrame()
    liew_train['smiles'] = liew_train_data['Canonical SMILES']
    liew_train['label'] = liew_train_data['Label']

    liew_test = pd.DataFrame()
    liew_test['smiles'] = liew_test_data['Canonical SMILES']
    liew_test['label'] = liew_test_data['Label']

    # liew_train.to_csv('./data/raw/liew_train.csv')
    # liew_test.to_csv('./data/raw/liew_test.csv')

    a = pd.concat([ncrt_train, liew_train]).reset_index(drop=True)
    a = a.drop_duplicates('smiles', keep='first')

    a.to_csv('./data/raw/ncrt_liew_train.csv')

    b = pd.concat([ncrt_test, liew_test]).reset_index(drop=True)
    b = b.drop_duplicates('smiles', keep='first')

    b.to_csv('./data/raw/ncrt_liew_test.csv')






    XU_validation = pd.read_excel(xls, 'supplementary table S1.4', header=1)

    XU_validation_data = pd.DataFrame()
    XU_validation_data['smiles'] = XU_validation['Canonical SMILES']
    XU_validation_data['label'] = XU_validation['Label']

    XU_validation_data.to_csv('./data/raw/xu_validation_data.csv')


    GREEN_validation = pd.read_excel(xls, 'supplementary table S1.3', header=1)

    GREEN_validation_data = pd.DataFrame()
    GREEN_validation_data['smiles'] = GREEN_validation['Canonical SMILES']
    GREEN_validation_data['label'] = GREEN_validation['Label']

    GREEN_validation_data.to_csv('./data/raw/green_validation_data.csv')
