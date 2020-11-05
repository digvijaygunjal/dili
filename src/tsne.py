import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from src.create_combined_fingerprints import update_columns
from src.create_tox21_fingerprints import convert_to_mol
from src.dnn_ncrt_whole import morgan_fingerprints, maccs_fingerprints, charge_features, autocorrelation_features, \
    harmonic_topology_index_feature


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
    train = pd.read_csv('./data/raw/ncrt_liew_train.csv', index_col=0).reindex()
    test = pd.read_csv('./data/raw/ncrt_liew_test.csv', index_col=0).reindex()

    data = pd.concat([train, test]).reset_index(drop=True).fillna(0)
    data = data.drop_duplicates('smiles', keep='first')

    dili_negative = data[data['label'] == 0].reset_index(drop=True)
    dili_positive = data[data['label'] == 1].reset_index(drop=True)

    fingerprints_neg = get_fingerprints(convert_to_mol(dili_negative['smiles']))
    fingerprints_pos = get_fingerprints(convert_to_mol(dili_positive['smiles']))

    common_comb = ['maccs', 'morgan', 'charge_f', 'harmonic_topology', 'autocorrelation_f']
    # common_comb = ['harmonic_topology']

    all_inputs_pos = pd.concat(list(map(lambda x: fingerprints_pos[x], common_comb)), axis=1).fillna(0)
    all_inputs_neg = pd.concat(list(map(lambda x: fingerprints_neg[x], common_comb)), axis=1).fillna(0)

    sns.set(rc={'figure.figsize': (5, 5)})
    palette = sns.color_palette("bright", 2)

    model = TSNE(metric="euclidean", n_components=2)
    X_embedded = model.fit_transform(all_inputs_neg)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], legend='full', palette=palette)

    sns.set(rc={'figure.figsize': (5, 5)})
    palette = sns.color_palette("bright", 2)

    model = TSNE(metric="euclidean", n_components=2)
    X_embedded = model.fit_transform(all_inputs_pos)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], legend='full', palette=palette, estimator=2)
