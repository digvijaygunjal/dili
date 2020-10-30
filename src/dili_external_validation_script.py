import argparse
import os
import os.path as op
import pickle
import tensorflow as tf

import numpy as np
from PyBioMed.Pymolecule import geary, charge, moran, topology
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import pandas as pd

from src.dnn_ncrt_liew_combined import calculate_scores, get_fingerprints

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('INFO')
tf.logging.set_verbosity(tf.logging.ERROR)


def load_model(file):
    # load model from pickle file
    if op.exists(file) and op.isfile(file):
        model = pickle.load(open(file, "rb"))
    else:
        print("File does not exist or is corrupted")
        return ()
    return (model)


def update_columns(data, prefix):
    data.columns = map(lambda c: prefix + str(c), data.columns)
    return data


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


def arg_parser():
    parser = argparse.ArgumentParser(description='DILI PREDICTION')
    parser.add_argument('--smiles', help='SMILES', default="")
    return parser.parse_args()


if __name__ == '__main__':
    file_path = "data/dnn_final.pkl"  # path to model file
    test_path = "data/raw/green_validation_data.csv"  # path to model file
    model = load_model(file_path)

    test = pd.read_csv(test_path, index_col=0).fillna(0).reset_index(drop=True)
    y_true = test['label']
    common_comb = ['maccs', 'morgan', 'charge_f', 'harmonic_topology', 'autocorrelation_f']

    mol = convert_to_mol(test['smiles'])
    fingerprints = get_fingerprints(mol)
    all_inputs = pd.concat(list(map(lambda x: fingerprints[x], common_comb)), axis=1).fillna(0)

    predicted = model.predict(all_inputs)
    predicted = list(np.round(predicted))
    predicted = list(map(lambda row: list(map(int, row)), predicted))
    score = calculate_scores(y_true, predicted)
    print(score)


# {'accuracy_score': 0.7030812324929971, 'hamming_loss': 0.2969187675070028, 'mcc': 0.3748847664786392, 'f1_score': 0.7809917355371903, 'roc_auc_score': 0.6608843537414966, 'balanced_accuracy_score': 0.6608843537414966, 'specificity': 0.4217687074829932, 'sensitivity': 0.9}
