import itertools
import pickle
from collections import Counter
from itertools import combinations

import keras
import numpy as np
import pandas as pd
from PyBioMed.Pymolecule import constitution, estate, moe, bcut, connectivity, molproperty, geary, charge, moran, \
    topology, cats2d
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

from src.create_combined_fingerprints import update_columns
from src.create_tox21_fingerprints import convert_to_mol
from src.dnn_ncrt import dnn
from src.rfc import classify_and_predict
from src.scoring import specificity, sensitivity
from imblearn.over_sampling import SMOTE


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


def morgan_fingerprints(mol_vector):
    morgan_fingerprints_vector = []
    for smiles in np.arange(len(mol_vector)):
        if not mol_vector[smiles]:
            morgan_fingerprints_vector.append([])
        else:
            fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol_vector[smiles], 2)
            morgan_fingerprints_vector.append(np.reshape(fingerprints, (1, -1))[0])
    return pd.DataFrame(morgan_fingerprints_vector).fillna(0)


def maccs_fingerprints(mol_vector):
    morgan_fingerprints_vector = []
    for smiles in np.arange(len(mol_vector)):
        if not mol_vector[smiles]:
            morgan_fingerprints_vector.append([])
        else:
            fingerprints = MACCSkeys.GenMACCSKeys(mol_vector[smiles])
            morgan_fingerprints_vector.append(np.reshape(fingerprints, (1, -1))[0])
    return pd.DataFrame(morgan_fingerprints_vector).fillna(0)


def charge_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 25 charge features
    """
    charge_vector = []
    error_examples = []
    for smiles in np.arange(len(mol_vector)):
        try:
            features = charge.GetCharge(mol_vector[smiles])
            charge_vector.append(features)
        except:
            error_examples.append(smiles)
            charge_vector.append(
                charge.GetCharge(mol_vector[smiles - 2]).fromkeys(charge.GetCharge(mol_vector[smiles - 2]), 0))
            pass
    return (pd.DataFrame.from_dict(charge_vector), error_examples)


def autocorrelation_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 64 autocorrelation features
    """
    moran_vector = []
    geary_vector = []
    error_examples = []
    for smiles in np.arange(len(mol_vector)):
        try:
            moran_features = moran.GetMoranAuto(mol_vector[smiles])
            geary_features = geary.GetGearyAuto(mol_vector[smiles])
            moran_vector.append(moran_features)
            geary_vector.append(geary_features)
        except:
            error_examples.append(smiles)
            moran_vector.append(
                moran.GetMoranAuto(mol_vector[smiles - 2]).fromkeys(moran.GetMoranAuto(mol_vector[smiles - 2]), 0))
            geary_vector.append(
                geary.GetGearyAuto(mol_vector[smiles - 2]).fromkeys(geary.GetGearyAuto(mol_vector[smiles - 2]), 0))
            pass
    moran_features_dataframe = pd.DataFrame.from_dict(moran_vector)
    geary_features_dataframe = pd.DataFrame.from_dict(geary_vector)
    return (pd.concat([moran_features_dataframe, geary_features_dataframe], axis=1), error_examples)


def harmonic_topology_index_feature(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : list containing " harmonic topological index " feature
    """
    harmonic_topology_index_feature_vector = []
    for smiles in np.arange(len(mol_vector)):
        try:
            features = topology.CalculateHarmonicTopoIndex(mol_vector[smiles])
            harmonic_topology_index_feature_vector.append(features)
        except:
            harmonic_topology_index_feature_vector.append(0)
            pass
    return (harmonic_topology_index_feature_vector)


def constitutional_features(mol_vector):
    constitutional_vector = []
    for smiles in np.arange(len(mol_vector)):
        features = constitution.GetConstitutional(mol_vector[smiles])
        constitutional_vector.append(features)
    return (pd.DataFrame.from_dict(constitutional_vector))


def topology_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 24 topological features
    """
    harmonic_topology_index = harmonic_topology_index_feature(mol_vector)
    topology_features_vector = []
    for smiles in np.arange(len(mol_vector)):
        GeometricTopoIndex = topology.CalculateGeometricTopoIndex(mol_vector[smiles])
        Balaban = topology.CalculateBalaban(mol_vector[smiles])
        ArithmeticTopoIndex = topology.CalculateArithmeticTopoIndex(mol_vector[smiles])
        BertzCT = topology.CalculateBertzCT(mol_vector[smiles])
        Diameter = topology.CalculateDiameter(mol_vector[smiles])
        GutmanTopo = topology.CalculateGutmanTopo(mol_vector[smiles])
        Harary = topology.CalculateHarary(mol_vector[smiles])
        Ipc = topology.CalculateIpc(mol_vector[smiles])
        MeanWeiner = topology.CalculateMeanWeiner(mol_vector[smiles])
        MZagreb1 = topology.CalculateMZagreb1(mol_vector[smiles])
        MZagreb2 = topology.CalculateMZagreb2(mol_vector[smiles])
        Petitjean = topology.CalculatePetitjean(mol_vector[smiles])
        Platt = topology.CalculatePlatt(mol_vector[smiles])
        PoglianiIndex = topology.CalculatePoglianiIndex(mol_vector[smiles])
        PolarityNumber = topology.CalculatePolarityNumber(mol_vector[smiles])
        Quadratic = topology.CalculateQuadratic(mol_vector[smiles])
        Radius = topology.CalculateRadius(mol_vector[smiles])
        Schiultz = topology.CalculateSchiultz(mol_vector[smiles])
        SimpleTopoIndex = topology.CalculateSimpleTopoIndex(mol_vector[smiles])
        Weiner = topology.CalculateWeiner(mol_vector[smiles])
        XuIndex = topology.CalculateXuIndex(mol_vector[smiles])
        Zagreb1 = topology.CalculateZagreb1(mol_vector[smiles])
        Zagreb2 = topology.CalculateZagreb2(mol_vector[smiles])
        topology_features_dictonary = {'Geto': GeometricTopoIndex, 'J': Balaban, 'Arto': ArithmeticTopoIndex,
                                       'BertzCT': BertzCT,
                                       'diametert': Diameter, 'GMTI': GutmanTopo, 'Thara': Harary, 'Ipc': Ipc,
                                       'AW': MeanWeiner,
                                       'MZM1': MZagreb1, 'MZM2': MZagreb2, 'petitjeant': Petitjean, 'Platt': Platt,
                                       'Dz': PoglianiIndex, 'Pol': PolarityNumber, 'Qindex': Quadratic,
                                       'radiust': Radius, 'Tsch': Schiultz,
                                       'Sito': SimpleTopoIndex, 'W': Weiner, 'Xu': XuIndex, 'ZM1': Zagreb1,
                                       'ZM2': Zagreb2,
                                       'Hato': harmonic_topology_index[smiles], }
        topology_features_vector.append(topology_features_dictonary)
    return (pd.DataFrame.from_dict(topology_features_vector))


def cats2d_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 150 cats2d descriptors
    """
    cats2d_vector = []
    for smiles in np.arange(len(mol_vector)):
        cats2d_features = cats2d.CATS2D(mol_vector[smiles])
        cats2d_vector.append(cats2d_features)
    return (pd.DataFrame.from_dict(cats2d_vector))


def kappa_descriptors(mol_vector):
    kappa_vector = []
    for smiles in np.arange(len(mol_vector)):
        kappa1 = Chem.rdMolDescriptors.CalcKappa1(mol_vector[smiles])
        kappa2 = Chem.rdMolDescriptors.CalcKappa2(mol_vector[smiles])
        kappa3 = Chem.rdMolDescriptors.CalcKappa3(mol_vector[smiles])
        features = [kappa1, kappa2, kappa3]
        kappa_vector.append(features)
    return (pd.DataFrame(kappa_vector))


def estate_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 316 estate features
    """
    estate_vector = []
    for smiles in np.arange(len(mol_vector)):
        features = estate.GetEstate(mol_vector[smiles])
        estate_vector.append(features)
    return (pd.DataFrame.from_dict(estate_vector))


def moe_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 59 moe features
    """
    moe_vector = []
    for smiles in np.arange(len(mol_vector)):
        features = moe.GetMOE(mol_vector[smiles])
        moe_vector.append(features)
    return (pd.DataFrame.from_dict(moe_vector))


def bcut_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 64 burden features
    """
    bcut_vector = []
    error_examples = []
    for smiles in np.arange(len(mol_vector)):
        try:
            features = bcut.GetBurden(mol_vector[smiles])
            bcut_vector.append(features)
        except:
            error_examples.append(smiles)
            bcut_vector.append(
                bcut.GetBurden(mol_vector[smiles - 2]).fromkeys(bcut.GetBurden(mol_vector[smiles - 2]), 0))
            pass
    return (pd.DataFrame.from_dict(bcut_vector), error_examples)


def connectivity_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 44 connectivity features
    """
    connectivity_vector = []
    for smiles in np.arange(len(mol_vector)):
        features = connectivity.GetConnectivity(mol_vector[smiles])
        connectivity_vector.append(features)
    return (pd.DataFrame.from_dict(connectivity_vector))


def molproperty_features(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 6 molproperty features
    """
    molproperty_vector = []
    error_examples = []
    for smiles in np.arange(len(mol_vector)):
        try:
            features = molproperty.GetMolecularProperty(mol_vector[smiles])
            molproperty_vector.append(features)
        except:
            error_examples.append(smiles)
            molproperty_vector.append(molproperty.GetMolecularProperty(mol_vector[smiles - 1]).fromkeys(
                molproperty.GetMolecularProperty(mol_vector[smiles - 1]), 0))
            pass
    return (pd.DataFrame.from_dict(molproperty_vector), error_examples)


def create_2D_descriptors(mol_vector):
    """
    param mol_vector : list of molecules in mol format
    return : dataset containing 2D descriptors
    """
    constitutional_features_dataframe = constitutional_features(mol_vector)
    estate_features_dataframe = estate_features(mol_vector)
    moe_features_dataframe = moe_features(mol_vector)
    bcut_features_dataframe, bcut_error_example = bcut_features(mol_vector)
    connectivity_features_dataframe = connectivity_features(mol_vector)
    molproperty_features_dataframe, molproperty_error_example = molproperty_features(mol_vector)
    charge_features_dataframe, charge_error_example = charge_features(mol_vector)
    autocorrelation_features_dataframe, autocorrelation_error_example = autocorrelation_features(mol_vector)
    topology_features_dataframe = topology_features(mol_vector)
    cats2d_features_dataframe = cats2d_features(mol_vector)
    kappa_descriptors_dataframe = kappa_descriptors(mol_vector)
    dataframe_list = [constitutional_features_dataframe, estate_features_dataframe,
                      moe_features_dataframe, bcut_features_dataframe, connectivity_features_dataframe,
                      molproperty_features_dataframe, charge_features_dataframe, autocorrelation_features_dataframe,
                      topology_features_dataframe, cats2d_features_dataframe, kappa_descriptors_dataframe]
    dataset = pd.concat(dataframe_list, axis=1)
    return (dataset)


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


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


if __name__ == "__main__":
    # ncrt_dili_non_null = pd.read_csv('./data/raw/ncrt_dili_non_null_round.csv', index_col=0).fillna(0).reindex()
    # ncrt_dili_non_null = pd.read_csv('./data/raw/ncrt_liew_train.csv', index_col=0).fillna(0).reset_index(drop=True)
    # ncrt_dili_non_null['severity_class'] = ncrt_dili_non_null['severity_class'].apply(lambda x: x/10)
    # ncrt_dili_non_null = ncrt_dili_non_null.round()

    ncrt_train = pd.read_csv('./data/raw/ncrt_liew_train.csv', index_col=0).fillna(0).reset_index(drop=True)
    ncrt_test = pd.read_csv('./data/raw/ncrt_liew_test.csv', index_col=0).fillna(0).reset_index(drop=True)

    fingerprints = get_fingerprints(convert_to_mol(ncrt_train['smiles']))
    fingerprints_test = get_fingerprints(convert_to_mol(ncrt_test['smiles']))

    common_comb = ['maccs', 'morgan', 'charge', 'harmonic_topology', 'autocorrelation', 'constitutional', 'estate',
                   'moe', 'bcut', 'molproperty', 'cats2d', 'kappa']


    y = pd.concat([ncrt_train['label'], ncrt_test['label']]).reset_index(drop=True)

    comb = []
    for i in range(1, len(common_comb)):
        comb.append(list(combinations(common_comb, i)))

    # dnn
    scores = []
    models = []
    for i in range(0, len(comb)):
        for j in range(0, len(comb[i])):
            all_inputs = pd.concat(list(map(lambda x: fingerprints[x], comb[i][j])), axis=1)
            all_inputs_test = pd.concat(list(map(lambda x: fingerprints_test[x], common_comb)), axis=1)
            X = pd.concat([all_inputs, all_inputs_test]).replace([np.inf, -np.inf, np.nan], 0)

            x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)

            print('Original dataset shape %s' % Counter(y_train))
            sm = SMOTE(random_state=1)
            x_train, y_train = sm.fit_resample(np.array(x_train), y_train)
            print('Resampled dataset shape %s' % Counter(y_train))

            classifier = dnn(x_train.shape[1], pd.DataFrame(y_train).shape[1])
            predicted, model = dnn_train_and_predict(classifier, x_train, x_test, y_train,
                                                         batch_size=64, epochs=10,
                                                         verbose=0,
                                                         validation_split=0.2)
            # pickle.dump(classifier, open("data/dnn_ncrt_whole.pkl", 'wb'))
            score = calculate_scores(y_test, predicted)
            print(score)
            score['key'] = " ".join(comb[i][j])
            scores.append(score)
            models.append(model)

    # Random Forest
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

            classifier = RandomForestClassifier(n_jobs=10)
            predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
            score = calculate_scores(y_test, predicted)
            print(score)
            score['key'] = " ".join(comb[i][j])
            scores.append(score)

    # columns = ['accuracy_score', 'hamming_loss', 'mcc', 'f1_score', 'roc_auc_score', 'balanced_accuracy_score', 'specificity', 'sensitivity']

    s = pd.DataFrame(scores)
    s.to_csv('data/results/rfc_all_combinations_ncrt_round.csv')
