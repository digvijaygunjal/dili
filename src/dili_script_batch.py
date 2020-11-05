import argparse
import os.path as op
import pickle

import numpy as np
import pandas as pd
from PyBioMed.Pymolecule import geary, charge, moran, topology
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

from src.dnn_ncrt import calculate_scores


def load_model(file):
    # load model from pickle file
    if op.exists(file) and op.isfile(file):
        model = pickle.load(open(file, "rb"))
    else:
        print("File does not exist or is corrupted")
        return None
    return model


def update_columns(data, prefix):
    data.columns = map(lambda c: prefix + str(c), data.columns)
    return data


def arg_parser():
    parser = argparse.ArgumentParser(description='DILI PREDICTION')
    parser.add_argument('--input-file', help='CSV input file with column smiles', default="./test.csv")
    parser.add_argument('--model-path', help='path to pkl file', default="./dnn_final.pkl")
    return parser.parse_args()


def get_descriptors(smiles):
    print(smiles)
    morgan = list(np.reshape(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2), (1, -1))[0])
    maccs = list(np.reshape(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)), (1, -1))[0])
    charge_d = list(charge.GetCharge(Chem.MolFromSmiles(smiles)).values())
    harmonic_topology = list(np.reshape([topology.CalculateHarmonicTopoIndex(Chem.MolFromSmiles(smiles))], (1, -1))[0])
    moran_d = list(moran.GetMoranAuto(Chem.MolFromSmiles(smiles)).values())
    geary_d = list(geary.GetGearyAuto(Chem.MolFromSmiles(smiles)).values())
    return morgan + maccs + charge_d + harmonic_topology + moran_d + geary_d


if __name__ == '__main__':
    input_file = arg_parser().input_file
    file_path = arg_parser().model_path
    model = load_model(file_path)

    data = pd.read_csv(input_file, sep=',').reset_index(drop=True)
    input_data = pd.DataFrame(list(map(get_descriptors, data['smiles'])))

    predicted = model.predict(input_data)
    print("&&&&&&&&&&&&&&&&&&&")
    print(predicted)
    predicted = list(np.round(predicted))
    predicted = list(map(lambda row: list(map(int, row)), predicted))

    print("Prediction: ", predicted)
    proba = model.predict_proba(input_data)
    print("Predict Proba : ", proba)

    pred = (proba > .5).astype(int)
    score = calculate_scores(data['label'], pred)
    print(score)
