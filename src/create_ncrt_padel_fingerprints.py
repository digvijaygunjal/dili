import pandas as pd
from padelpy import from_smiles


def padel(data):
    padel_desc = []
    for i, smile in data.iterrows():
        csv = './data/intermediate/padel/' + str(i) + '.csv'
        features = from_smiles(smiles=smile['smiles'], fingerprints=True, timeout=20, descriptors=True, output_csv=csv)
        padel_desc.append(features)
    return (pd.DataFrame(padel_desc))


if __name__ == "__main__":
    train = pd.read_csv('./data/raw/ncrt_train.csv', index_col=0).reindex()
    test = pd.read_csv('./data/raw/ncrt_test.csv', index_col=0).reindex()
    test_len = len(test)
    train_len = len(train)

    data = pd.concat([train, test], ignore_index=True)
    padel_d = padel(data[364:])


    fingerprints = padel_d
    fingerprints['label'] = data['label']
    train_fingerprints = fingerprints[0:train_len]
    test_fingerprints = fingerprints[train_len:]
    fingerprints.to_csv('./data/intermediate/ncrt_fingerprints.csv')
    train_fingerprints.to_csv('./data/intermediate/ncrt_train_padel_fingerprints.csv')
    test_fingerprints.to_csv('./data/intermediate/ncrt_test_padel_fingerprints.csv')
