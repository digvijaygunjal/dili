import numpy as np
import pandas as pd
from mordred import Calculator, descriptors

from src.create_tox21_fingerprints import convert_to_mol, maccs_keys_fingerprints, morgan_fingerprints


def update_columns(data, prefix):
    data.columns = map(lambda c: prefix + str(c), data.columns)
    return data


def create_3D_descriptors(mol_vector):
    calculator = Calculator(descriptors, ignore_3D=False)
    descriptors_3d = []
    for smiles in np.arange(len(mol_vector)):
        if not mol_vector[smiles]:
            descriptors_3d.append({})
        else:
            features = calculator(mol_vector[smiles])[1613:]
            descriptors_3d.append(features)
    return (pd.DataFrame(descriptors_3d))


if __name__ == "__main__":
    train = pd.read_csv('./data/raw/ncrt_train.csv', index_col=0).reindex()
    test = pd.read_csv('./data/raw/ncrt_test.csv', index_col=0).reindex()
    test_len = len(test)
    train_len = len(train)

    data = pd.concat([train, test], ignore_index=True)
    mol = convert_to_mol(data['smiles'])
    maccs = update_columns(maccs_keys_fingerprints(mol), 'maccs')
    morgan = update_columns(morgan_fingerprints(mol), 'morgan')
    three_d = update_columns(create_3D_descriptors(mol), 'td')

    fingerprints = pd.concat([maccs, morgan, three_d], axis=1)
    fingerprints['label'] = data['label']
    train_fingerprints = fingerprints[0:train_len]
    test_fingerprints = fingerprints[train_len:]
    fingerprints.to_csv('./data/intermediate/ncrt_fingerprints.csv')
    train_fingerprints.to_csv('./data/intermediate/ncrt_train_fingerprints.csv')
    test_fingerprints.to_csv('./data/intermediate/ncrt_test_fingerprints.csv')
