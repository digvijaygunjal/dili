import pickle
from collections import Counter
from collections import Counter
from itertools import combinations

import keras
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, roc_auc_score, \
    balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef

from src.create_ncrt_fingerprints import update_columns
from src.create_tox21_fingerprints import convert_to_mol
from src.dnn_ncrt import dnn
from src.dnn_ncrt_whole import morgan_fingerprints, maccs_fingerprints, charge_features, autocorrelation_features, \
    harmonic_topology_index_feature
from src.rfc import classify_and_predict
from src.scoring import specificity, sensitivity


def calculate_scores(actual, predicted):
    return {
        'accuracy_score': accuracy_score(actual, predicted),
        'hamming_loss': hamming_loss(actual, predicted),
        'mcc': matthews_corrcoef(actual, predicted),
        'f1_score': f1_score(actual, predicted),
        'roc_auc_score': roc_auc_score(actual, predicted),
        'balanced_accuracy_score': balanced_accuracy_score(actual, predicted),
        'specificity': specificity(actual, predicted),
        'sensitivity': sensitivity(actual, predicted)
    }


def test_apply(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


def get_fingerprints(mol):
    morgan = update_columns(morgan_fingerprints(mol), 'morgan')
    maccs = update_columns(maccs_fingerprints(mol), 'maccs')
    charge_f, e = charge_features(mol)
    autocorrelation_f, e = autocorrelation_features(mol)
    harmonic_topology = pd.DataFrame(harmonic_topology_index_feature(mol))
    return {
        "maccs": maccs,
        "morgan": morgan,
        'charge_f': update_columns(charge_f, 'charge'),
        'autocorrelation_f': update_columns(autocorrelation_f, 'autocorrelation_f'),
        'harmonic_topology': update_columns(harmonic_topology, 'harmonic')
    }


if __name__ == "__main__":
    ncrt_train = pd.read_csv('./data/raw/ncrt_liew_train.csv', index_col=0).fillna(0).reset_index(drop=True)
    ncrt_test = pd.read_csv('./data/raw/ncrt_liew_test.csv', index_col=0).fillna(0).reset_index(drop=True)

    # data = pd.concat([ncrt_train, ncrt_test])
    # data = data.drop_duplicates('smiles', keep='first').reset_index(drop=True)
    # data.to_csv("./data/raw/ncrt_liew_combined.csv")

    common_comb = ['maccs', 'morgan', 'charge_f', 'harmonic_topology']

    fingerprints = get_fingerprints(convert_to_mol(ncrt_train['smiles']))
    fingerprints_test = get_fingerprints(convert_to_mol(ncrt_test['smiles']))

    comb = []
    for i in range(1, len(fingerprints.keys()) + 1):
        comb.append(list(combinations(fingerprints.keys(), i)))

    # rfc
    scores = []
    # for i in range(0, len(comb)):
    #     for j in range(0, len(comb[i])):
    all_inputs = pd.concat(list(map(lambda x: fingerprints[x], common_comb)), axis=1).fillna(0)
    all_inputs_test = pd.concat(list(map(lambda x: fingerprints_test[x], common_comb)), axis=1).fillna(0)

    x_train = all_inputs
    x_test = all_inputs_test
    y_train = ncrt_train['label']
    y_test = ncrt_test['label']

    print('Original dataset shape %s' % Counter(y_train))
    sm = SMOTE(random_state=1)
    x_train, y_train = sm.fit_resample(np.array(x_train), y_train)
    print('Resampled dataset shape %s' % Counter(y_train))

    classifier = RandomForestClassifier(n_jobs=10)
    predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
    score = calculate_scores(y_test, predicted)
    print(score)
    score['key'] = " ".join(common_comb)
    scores.append(score)
    s = pd.DataFrame(scores)
    s.to_csv('data/results/rfc_ncrt_liew.csv')
