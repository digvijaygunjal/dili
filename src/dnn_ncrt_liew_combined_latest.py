from collections import Counter
from itertools import combinations

import keras
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from keras import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, roc_auc_score, \
    balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from src.create_ncrt_fingerprints import update_columns
from src.create_tox21_fingerprints import convert_to_mol
from src.dnn_ncrt import dnn
from src.dnn_ncrt_whole import morgan_fingerprints, maccs_fingerprints, charge_features, autocorrelation_features, \
    harmonic_topology_index_feature, dnn3d, constitutional_features, topology_features, create_2D_descriptors, \
    estate_features, moe_features, bcut_features, connectivity_features, molproperty_features, cats2d_features, \
    kappa_descriptors
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

def dnn_train_and_predict(classifier, X_train, X_test, y_train, batch_size=64, epochs=50, verbose=0,
                          validation_split=0.2):
    classifier.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=['accuracy'],
    )
    classifier.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                   validation_split=validation_split)
    predicted = classifier.predict(X_test)
    predicted = list(np.round(predicted))
    predicted = list(map(lambda row: list(map(int, row)), predicted))
    return predicted, classifier


def get_fingerprints(mol):
    morgan = update_columns(morgan_fingerprints(mol), 'morgan')
    maccs = update_columns(maccs_fingerprints(mol), 'maccs')
    charge_f, e = charge_features(mol)
    autocorrelation_f, e = autocorrelation_features(mol)
    harmonic_topology = pd.DataFrame(harmonic_topology_index_feature(mol))
    constitutional_f = constitutional_features(mol)
    estate_f = estate_features(mol)
    moe_f = moe_features(mol)
    bcut_f, e = bcut_features(mol)
    molproperty_f, e = molproperty_features(mol)
    cats2d_f = cats2d_features(mol)
    kappa = kappa_descriptors(mol)

    return {
        "maccs": maccs,
        "morgan": morgan,
        'charge': update_columns(charge_f, 'charge'),
        'harmonic_topology': update_columns(harmonic_topology, 'harmonic'),
        'autocorrelation': update_columns(autocorrelation_f, 'autocorrelation'),
        'constitutional': update_columns(constitutional_f, 'constitutional'),
        'estate': update_columns(estate_f, 'estate'),
        'moe': update_columns(moe_f, 'moe'),
        'bcut': update_columns(bcut_f, 'bcut'),
        'molproperty': update_columns(molproperty_f, 'molproperty'),
        'cats2d': update_columns(cats2d_f, 'cats2d'),
        'kappa': update_columns(kappa, 'kappa'),
    }


if __name__ == "__main__":
    ncrt_train = pd.read_csv('./data/raw/ncrt_liew_train.csv', index_col=0).fillna(0).reset_index(drop=True)
    ncrt_test = pd.read_csv('./data/raw/ncrt_liew_test.csv', index_col=0).fillna(0).reset_index(drop=True)

    common_comb = ['maccs', 'morgan', 'charge', 'harmonic_topology', 'autocorrelation', 'constitutional', 'estate',
                   'moe', 'bcut', 'molproperty', 'cats2d', 'kappa']

    fingerprints = get_fingerprints(convert_to_mol(ncrt_train['smiles']))
    fingerprints_test = get_fingerprints(convert_to_mol(ncrt_test['smiles']))

    all_inputs = pd.concat(list(map(lambda x: fingerprints[x], common_comb)), axis=1).fillna(0)
    all_inputs_test = pd.concat(list(map(lambda x: fingerprints_test[x], common_comb)), axis=1).fillna(0)

    X = pd.concat([all_inputs, all_inputs_test]).reset_index(drop=True)
    y = pd.concat([ncrt_train['label'], ncrt_test['label']]).reset_index(drop=True)

    # Instantiate the cross validator
    skf = StratifiedKFold(n_splits=10, random_state=1)

    scores = []
    models = []
    for index, (train_indices, val_indices) in list(enumerate(skf.split(X, y))):
        print("Fold " + str(index + 1) + "/10...")

        x_train, xval = X.iloc[train_indices], X.iloc[val_indices]
        y_train, yval = y.iloc[train_indices], y.iloc[val_indices]

        print('Original dataset shape %s' % Counter(y_train))
        sm = SMOTE(random_state=1)
        x_train, y_train = sm.fit_resample(np.array(x_train), y_train)
        print('Resampled dataset shape %s' % Counter(y_train))

        # Clear model, and create it DNN
        model = dnn(x_train.shape[1], pd.DataFrame(y_train).shape[1])

        predicted, model = dnn_train_and_predict(model, x_train, xval, y_train,
                                                     batch_size=64, epochs=10,
                                                     verbose=0,
                                                     validation_split=0.2)

        # RFC
        # classifier = RandomForestClassifier(n_jobs=10)
        # predicted, model = classify_and_predict(x_train, xval, y_train, classifier)

        score = calculate_scores(yval, predicted)
        print(score)
        scores.append(score)
        models.append(model)

    pd.DataFrame(scores).to_csv("./data/results/kfold_ncrt_liew.csv")
