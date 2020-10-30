import itertools
import pickle
from collections import Counter
from itertools import combinations

import keras
import numpy as np
import pandas as pd
from PyBioMed.Pymolecule import constitution, estate, moe, bcut, connectivity, molproperty, geary, charge, moran, \
    topology, cats2d
from imblearn.over_sampling import SMOTE
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout
from keras.models import Sequential
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, roc_auc_score, \
    balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from src.create_tox21_fingerprints import convert_to_mol
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


def dnn(input_shape, output_size):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(input_shape,), kernel_initializer=glorot_normal(),
                    bias_initializer=glorot_normal()))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.2))
    model.add(
        Dense(output_size, activation="sigmoid", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    return keras.models.clone_model(model)


def dnn3d(input_shape, output_size):
    model = Sequential()
    model.add(Dense(1000, activation="relu", input_shape=(input_shape,), kernel_initializer=glorot_normal(),
                    bias_initializer=glorot_normal()))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.3))
    model.add(Dense(300, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.3))
    model.add(Dense(150, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.3))
    model.add(Dense(70, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    model.add(Dropout(0.2))
    model.add(
        Dense(output_size, activation="sigmoid", kernel_initializer=glorot_normal(), bias_initializer=glorot_normal()))
    return model


def dnn_train_and_predict(classifier, X_train, X_test, y_train, batch_size=64, epochs=50, verbose=0,
                          validation_split=0.2):
    classifier.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    classifier.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                   validation_split=validation_split)
    predicted = classifier.predict(X_test)
    predicted = list(np.round(predicted))
    predicted = list(map(lambda row: list(map(int, row)), predicted))
    classifier.save("./data/dnn_binary_filtered")
    return predicted, classifier


def test_apply(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


def get_fingerprints(mol):
    morgan = morgan_fingerprints(mol)
    maccs = maccs_fingerprints(mol)
    charge_f, e = charge_features(mol)
    autocorrelation_f, e = autocorrelation_features(mol)
    harmonic_topology = pd.DataFrame(harmonic_topology_index_feature(mol))
    return {
        "maccs": maccs,
        "morgan": morgan,
        'charge_f': charge_f,
        'autocorrelation_f': autocorrelation_f,
        'harmonic_topology': harmonic_topology
    }


if __name__ == "__main__":
    ncrt_train = pd.read_csv('./data/raw/ncrt_train.csv', index_col=0).fillna(0).reindex()
    ncrt_test = pd.read_csv('./data/raw/ncrt_test.csv', index_col=0).fillna(0).reindex()

    data = pd.concat([ncrt_train, ncrt_test])
    fingerprints = get_fingerprints(convert_to_mol(data['smiles']))


    comb = []
    for i in range(1, len(fingerprints.keys())):
        comb.append(list(combinations(fingerprints.keys(), i)))


    # dnn
    scores = []
    for i in range(0, len(comb)):
        for j in range(0, len(comb[i])):
            all_inputs = pd.concat(list(map(lambda x: fingerprints[x], comb[i][j])), axis=1).fillna(0)

            x_train, x_test, y_train, y_test = train_test_split(all_inputs, ncrt_dili_non_null['severity_class'],
                                                                random_state=1)

            print('Original dataset shape %s' % Counter(y_train))
            sm = SMOTE(random_state=1)
            x_train, y_train = sm.fit_resample(np.array(x_train), y_train)
            print('Resampled dataset shape %s' % Counter(y_train))

            classifier = dnn(x_train.shape[1], pd.DataFrame(y_train).shape[1])
            predicted, classfier = dnn_train_and_predict(classifier, x_train.values, x_test.values, y_train.values,
                                                         batch_size=64, epochs=10,
                                                         verbose=0,
                                                         validation_split=0.2)
            # pickle.dump(classifier, open("data/dnn_ncrt_whole.pkl", 'wb'))
            score = calculate_scores(y_test, predicted)
            print(score)
            score['key'] = " ".join(comb[i][j])
            scores.append(score)

    # Random Forest
    scores = []
    for i in range(0, len(comb)):
        for j in range(0, len(comb[i])):
            all_inputs = pd.concat(list(map(lambda x: fingerprints[x], comb[i][j])), axis=1).fillna(0)

            x_train, x_test, y_train, y_test = train_test_split(all_inputs, data['label'],
                                                                random_state=1)

            print('Original dataset shape %s' % Counter(y_train))
            sm = SMOTE(random_state=1)
            x_train, y_train = sm.fit_resample(np.array(x_train), y_train)
            print('Resampled dataset shape %s' % Counter(y_train))

            classifier = RandomForestClassifier(n_jobs=10)
            predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
            score = calculate_scores(y_test, predicted)
            print(score)
            score['key'] = " ".join(comb[i][j])
            scores.append(score)

    # columns = ['accuracy_score', 'hamming_loss', 'mcc', 'f1_score', 'roc_auc_score', 'balanced_accuracy_score', 'specificity', 'sensitivity']

    s = pd.DataFrame(scores)
    s.to_csv('data/results/rfc_all_combinations_ncrt_round.csv')
