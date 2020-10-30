import csv
import pickle
import os.path as op
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Torsions


#################################TOCOSTUMIZE#################################


def load_model(file):
    #load model from pickle file
    if op.exists(file) and op.isfile(file):
        model = pickle.load(open(file, "rb"))
    else:
        print("File does not exist or is corrupted")
        return()
    return(model)

def create_descriptor(smiles, choice):
     #creates fingerprint for given SMILES
    m = AllChem.MolFromSmiles(smiles)
    if choice == 1:
        descriptor = AllChem.GetMorganFingerprintAsBitVect(m,2)
    elif choice == 2:
        descriptor = MACCSkeys.GenMACCSKeys(m)
    elif choice == 3:
        descriptor = Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m)
    else:
        print ('Invalid fingerprint choice')
    return(descriptor)


if __name__ == '__main__':
    file_path = "data/rfc_ncrt.pkl"  # path to model file
    smiles = "Clc(c(Cl)c(Cl)c1C(=O)O)c(Cl)c1Cl"  # SMILES to classify/ or a csv file (containing SMILES)
    fp_type = 3  # must match the on in the model: 1 == Morgan, 2 == Maccs, 3 == Topological Torsions (or the features you have used in your model)

    #perform classification for a given SMILES with a saved model

    model = load_model(file_path)
    fp1 = create_descriptor('CC(=O)Oc1ccc(cc1)C(=C2CCCCC2)c3ccc(cc3)OC(=O)C', 1)
    fp2 = create_descriptor('CC(=O)Oc1ccc(cc1)C(=C2CCCCC2)c3ccc(cc3)OC(=O)C', 2)
    target = np.reshape(fp1, (1, -1))
    pred = model.predict(target)
    print(pred)