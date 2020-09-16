import numpy as np
import pandas as pd


def get_values_for_molecules(molecules, train_dataset, value_index=2, default_value=None):
    train_values = []
    for m in molecules:
        values = train_dataset.loc[train_dataset[0] == m].values
        if len(values) > 0:
            train_values.append(values[0][value_index])
        else:
            train_values.append(default_value)

    return train_values


train_data_files = {
    "nr-ahr": "./data/raw/tox21smiles/nr-ahr.smiles",
    "nr-er-lbd": "./data/raw/tox21smiles/nr-er-lbd.smiles",
    "sr-hse": "./data/raw/tox21smiles/sr-hse.smiles"
}
if __name__ == "__main__":
    nr_ahr = pd.read_csv(train_data_files["nr-ahr"], sep="\t", header=None)
    nr_er_lbd = pd.read_csv(train_data_files["nr-er-lbd"], sep="\t", header=None)
    sr_hse = pd.read_csv(train_data_files["sr-hse"], sep="\t", header=None)

    molecules = np.unique(pd.concat([nr_ahr[0], nr_er_lbd[0], sr_hse[0]]))

    train = pd.DataFrame()
    train['molecule'] = molecules
    train['NR.ahr'] = get_values_for_molecules(molecules, nr_ahr)
    train['NR.erlbd'] = get_values_for_molecules(molecules, nr_er_lbd)
    train['SR.hse'] = get_values_for_molecules(molecules, sr_hse)

    train.to_csv('./data/intermediate/tox21_train_3_labels.csv')
