import argparse
import os.path as op
import pickle

import numpy as np
from PyBioMed.Pymolecule import geary, charge, moran, topology
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


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
    parser.add_argument('--smiles', help='SMILES', default="c1cc(cnc1)C(=O)O")
    parser.add_argument('--model-path', help='Path of the pickle file', default="./dnn_final.pkl")
    return parser.parse_args()


if __name__ == '__main__':
    smiles = arg_parser().smiles
    file_path = arg_parser().model_path
    model = load_model(file_path)

    morgan = list(np.reshape(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2), (1, -1))[0])
    maccs = list(np.reshape(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)), (1, -1))[0])
    _charge = list(charge.GetCharge(Chem.MolFromSmiles(smiles)).values())
    harmonic_topology = list(np.reshape([topology.CalculateHarmonicTopoIndex(Chem.MolFromSmiles(smiles))], (1, -1))[0])
    _moran = list(moran.GetMoranAuto(Chem.MolFromSmiles(smiles)).values())
    _geary = list(geary.GetGearyAuto(Chem.MolFromSmiles(smiles)).values())

    fp = morgan + maccs + _charge + harmonic_topology + _moran + _geary

    target = np.reshape(fp, (1, -1))
    pred = model.predict(target)
    print("Prediction: ", np.round(pred))
    print("Predict Proba : ", model.predict_proba(target))
