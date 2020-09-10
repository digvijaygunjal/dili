import numpy as np
import pandas as pd
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
            maccs_keys_fingerprints_vector.append(0)
        else:
            fingerprints = MACCSkeys.GenMACCSKeys(mol_vector[smiles])
            all_fingerprints.append(fingerprints)
            maccs_keys_fingerprints_vector.append(fingerprints[1])
    maccs_keys_fingerprints_df = pd.DataFrame.from_dict(maccs_keys_fingerprints_vector)
    maccs_keys_fingerprints_df = maccs_keys_fingerprints_df.fillna(0)
    return (pd.DataFrame(maccs_keys_fingerprints_df))


def morgan_fingerprints(mol_vector):
    morgan_fingerprints_vector = []
    for smiles in np.arange(len(mol_vector)):
        if not mol_vector[smiles]:
            morgan_fingerprints_vector.append(0)
        else:
            fingerprints = AllChem.GetMorganFingerprint(mol_vector[smiles], 2)
            morgan_fingerprints_vector.append(fingerprints[1])
    morgan_fingerprints_df = pd.DataFrame.from_dict(morgan_fingerprints_vector)
    morgan_fingerprints_df = morgan_fingerprints_df.fillna(0)
    return (pd.DataFrame(morgan_fingerprints_df))


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


training_dataset_file_path = "./data/raw/tox21_10k_data_all.sdf"

train_data_files = {
    "nr-ahr": "./data/raw/tox21smiles/nr-ahr.smiles",
    "nr-er-lbd": "./data/raw/tox21smiles/nr-er-lbd.smiles",
    "sr-hse": "./data/raw/tox21smiles/sr-hse.smiles"
}

smiles_data_train = sdf_to_smiles(training_dataset_file_path)
nr_ahr = pd.read_csv(train_data_files["nr-ahr"], sep="\t", header=None)
nr_er_lbd = pd.read_csv(train_data_files["nr-er-lbd"], sep="\t", header=None)
sr_hse = pd.read_csv(train_data_files["sr-hse"], sep="\t", header=None)

# m = Chem.MolFromSmiles('[NH4+].[NH4+].F[Si--](F)(F)(F)(F)F')
# Descriptors.TPSA(m)

nr_ahr_mol = convert_to_mol(nr_ahr[0])
maccs_train_data = maccs_keys_fingerprints(nr_ahr_mol)
morgan_train_data = morgan_fingerprints(nr_ahr_mol)

train_data.to_csv('./data/intermediate/tox21.csv')
