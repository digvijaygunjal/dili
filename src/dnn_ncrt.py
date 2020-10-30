import pickle

import keras
import numpy as np
import pandas as pd
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, roc_auc_score, \
    balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
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


def convert_to_float(data):
    new_data = pd.DataFrame(data)
    for column in new_data.columns:
        new_data[column] = new_data[column].apply(test_apply)
    return new_data


if __name__ == "__main__":
    train = pd.read_csv('./data/intermediate/ncrt_train_maccs_morgan_fingerprints.csv', index_col=0).fillna(0).reindex()
    test = pd.read_csv('./data/intermediate/ncrt_test_maccs_morgan_fingerprints.csv', index_col=0).fillna(0).reindex()

    x_train = train.iloc[:, :-1].iloc[:, 1:]
    x_test = test.iloc[:, :-1].iloc[:, 1:]
    y_train = train.iloc[:, -1:]
    y_test = test.iloc[:, -1:]

    x_train.columns = x_train.columns.str.strip()
    x_test.columns = x_test.columns.str.strip()
    y_train.columns = y_train.columns.str.strip()
    y_train.columns = y_train.columns.str.strip()

    x_train = convert_to_float(x_train)
    x_test = convert_to_float(x_test)
    y_train = convert_to_float(y_train)
    y_test = convert_to_float(y_test)

    classifier = dnn3d(x_train.shape[1], pd.DataFrame(y_train).shape[1])
    predicted, classfier = dnn_train_and_predict(classifier, x_train.values, x_test.values, y_train.values,
                                                 batch_size=64, epochs=10,
                                                 verbose=0,
                                                 validation_split=0.2)
    pickle.dump(classifier, open("data/dnn_ncrt", 'wb'))
    scores = calculate_scores(y_test, predicted)
    print(scores)
