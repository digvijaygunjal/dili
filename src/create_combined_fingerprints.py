import pandas as pd

from src.create_tox21_fingerprints import convert_to_mol, maccs_keys_fingerprints, morgan_fingerprints


def update_columns(data, prefix):
    data.columns = map(lambda c: prefix + str(c), data.columns)
    return data


if __name__ == "__main__":
    train = pd.read_csv('./data/raw/combined_train.csv', index_col=0).reindex()
    test = pd.read_csv('./data/raw/combined_test.csv', index_col=0).reindex()
    test_len = len(test)
    train_len = len(train)

    data = pd.concat([train, test], ignore_index=True)
    mol = convert_to_mol(data['smiles'])
    maccs = update_columns(maccs_keys_fingerprints(mol), 'maccs')
    morgan = update_columns(morgan_fingerprints(mol), 'morgan')

    fingerprints = pd.concat([maccs, morgan], axis=1)
    fingerprints['label'] = data['label']
    train_fingerprints = fingerprints[0:train_len]
    test_fingerprints = fingerprints[train_len:]
    fingerprints.to_csv('./data/intermediate/combined_maccs_morgan_fingerprints.csv')
    train_fingerprints.to_csv('./data/intermediate/combined_train_maccs_morgan_fingerprints.csv')
    test_fingerprints.to_csv('./data/intermediate/combined_test_maccs_morgan_fingerprints.csv')
