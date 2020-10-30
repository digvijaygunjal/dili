import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from itertools import combinations
import xgboost as xgb

from sklearn.metrics import accuracy_score, hamming_loss, f1_score, roc_auc_score, \
    balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef

from src.create_ncrt_fingerprints import update_columns
from src.create_tox21_fingerprints import convert_to_mol
from src.dnn_ncrt_whole import morgan_fingerprints, maccs_fingerprints, charge_features, autocorrelation_features, \
    harmonic_topology_index_feature
from src.rfc import classify_and_predict
from src.scoring import specificity, sensitivity
from sklearn.model_selection import StratifiedKFold

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

    common_comb = ['maccs', 'morgan', 'charge_f', 'harmonic_topology', 'autocorrelation_f']

    fingerprints = get_fingerprints(convert_to_mol(ncrt_train['smiles']))
    fingerprints_test = get_fingerprints(convert_to_mol(ncrt_test['smiles']))

    # XG_boost
    scores = []
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
    print(common_comb)
    classifier = xgb.XGBClassifier()
    predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
    score = calculate_scores(y_test, predicted)
    pickle.dump(classifier, open("data/xg_boost.pkl", 'wb'))
    print(score)
    score['key'] = " ".join(common_comb)
    scores.append(score)

    s = pd.DataFrame(scores)
    s.to_csv('data/results/xg_boost_combined_all.csv')

# # Instantiate the cross validator
# skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
# # Loop through the indices the split() method returns
# for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
#     print
#     "Training on fold " + str(index + 1) + "/10..."
#     # Generate batches from indices
#     xtrain, xval = X[train_indices], X[val_indices]
#     ytrain, yval = y[train_indices], y[val_indices]
#     # Clear model, and create it
#     model = None
#     model = xgb.XGBClassifier()
#
#     # Debug message I guess
#     # print "Training new iteration on " + str(xtrain.shape[0]) + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while..."
#
#     history = train_model(model, xtrain, ytrain, xval, yval)
#     accuracy_history = history.history['acc']
#     val_accuracy_history = history.history['val_acc']
#     print("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(
#         val_accuracy_history[-1]))