import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    train = pd.read_csv('./data/intermediate/ncrt_train_maccs_morgan_fingerprints.csv', index_col=0).reindex()
    test = pd.read_csv('./data/intermediate/ncrt_test_maccs_morgan_fingerprints.csv', index_col=0).reindex()

    x_train = train.iloc[:, :-1]
    x_test = test.iloc[:, :-1]
    y_train = train['label']
    y_test = test['label']

    model = TSNE(metric="euclidean")
    x_train_tsne = model.fit_transform(x_train)
    df_subset = {'tsne-2d-one': x_train_tsne[:, 0], 'tsne-2d-two': x_train_tsne[:, 1]}

    df = pd.DataFrame(x_train)
    df['y'] = y_train
    df['label'] = df['y'].apply(lambda i: str(i))

    df_subset = df
    data_subset = df_subset.values

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
