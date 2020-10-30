import keras
import numpy as np
import pandas as pd
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, matthews_corrcoef, roc_auc_score, \
    balanced_accuracy_score

from src.dnn_ncrt import dnn
from src.rfc import calculate_scores
from src.scoring import specificity, sensitivity


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
    # classifier.save("./data/dnn_binary_filtered")
    return predicted, classifier


if __name__ == "__main__":
    train = pd.read_csv('./data/intermediate/combined_train_maccs_morgan_fingerprints.csv', index_col=0).reindex()
    test = pd.read_csv('./data/intermediate/combined_test_maccs_morgan_fingerprints.csv', index_col=0).reindex()

    x_train = train.iloc[:, :-1]
    x_test = test.iloc[:, :-1]
    y_train = train['label']
    y_test = test['label']

    classifier = dnn(x_train.shape[1], pd.DataFrame(y_train).shape[1])
    predicted, classfier = dnn_train_and_predict(classifier, x_train, x_test, y_train, batch_size=64, epochs=10,
                                                 verbose=0,
                                                 validation_split=0.2)
    scores = calculate_scores(y_test, predicted)
    print(scores)
