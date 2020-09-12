import numpy as np
import pandas as pd
from PyBioMed.PyMolecule import fingerprint
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.PandasTools import LoadSDF


def sdf_to_smiles(file_path):
    smiles_dataframe = LoadSDF(file_path, smilesName='SMILES')
    smiles_dataframe = smiles_dataframe.reset_index(drop=True)
    smiles_data = smiles_dataframe["SMILES"]
    return (smiles_data)


def maccs_keys_fingerprints(mol_vector):
    maccs_keys_fingerprints_vector = []
    all_fingerprints = []
    for smiles in np.arange(len(mol_vector)):
        if not mol_vector[smiles]:
            maccs_keys_fingerprints_vector.append({})
        else:
            fingerprints = fingerprint.CalculateMACCSFingerprint(mol_vector[smiles])
            all_fingerprints.append(fingerprints)
            maccs_keys_fingerprints_vector.append(fingerprints[1])
    maccs_keys_fingerprints_df = pd.DataFrame.from_dict(maccs_keys_fingerprints_vector)
    maccs_keys_fingerprints_df = maccs_keys_fingerprints_df.fillna(0)
    return pd.DataFrame(maccs_keys_fingerprints_df)


def morgan_fingerprints(mol_vector):
    morgan_fingerprints_vector = []
    for smiles in np.arange(len(mol_vector)):
        if not mol_vector[smiles]:
            morgan_fingerprints_vector.append({})
        else:
            fingerprints = fingerprint.CalculateMorganFingerprint(mol_vector[smiles], 2)
            morgan_fingerprints_vector.append(fingerprints[1])
    morgan_fingerprints_df = pd.DataFrame.from_dict(morgan_fingerprints_vector)
    morgan_fingerprints_df = morgan_fingerprints_df.fillna(0)
    return pd.DataFrame(morgan_fingerprints_df)


def convert_to_mol(smiles_data):
    """
    converts SMILES data in mol format
    param smiles_data : data in SMILES format
    return : returns data in mol format
    """
    mol_vector = []
    for smile in np.arange(len(smiles_data)):
        mol = Chem.MolFromSmiles(smiles_data[smile])
        mol_vector.append(mol)
    return (mol_vector)


train_data_files = {
    "nr-ahr": "./data/raw/tox21smiles/nr-ahr.smiles",
    "nr-er-lbd": "./data/raw/tox21smiles/nr-er-lbd.smiles",
    "sr-hse": "./data/raw/tox21smiles/sr-hse.smiles"
}

train = pd.read_csv('./data/intermediate/tox21_train_3_labels.csv', index_col=0).reindex()

mol = convert_to_mol(train['molecule'])
maccs_train_data = maccs_keys_fingerprints(mol)
morgan_train_data = morgan_fingerprints(mol)

fingerprints = pd.concat([maccs_train_data, morgan_train_data], axis=1)
fingerprints['NR.ahr'] = train['NR.ahr']
fingerprints['NR.erlbd'] = train['NR.erlbd']
fingerprints['SR.hse'] = train['SR.hse']

fingerprints.to_csv('./data/intermediate/tox21_maccs_morgan_fingerprints.csv')
