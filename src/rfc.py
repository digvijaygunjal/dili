import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split


def classify_and_predict(x_train, x_test, y_train, model):
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    return y_predicted, model


def calculate_scores(actual, predicted):
    return {
        'accuracy_score': (accuracy_score(actual, predicted)),
        'hamming_loss': (hamming_loss(actual, predicted)),
        'mcc': (matthews_corrcoef(actual, predicted)),
        'f1_score': (f1_score(actual, predicted))
    }


train = pd.read_csv('./data/intermediate/tox21_maccs_morgan_fingerprints.csv', index_col=0).reindex()

to_predict = pd.DataFrame()
to_predict['NR.ahr'] = train['NR.ahr']
to_predict = to_predict.fillna(0)
values = train.iloc[:, :-3]
x_train, x_test, y_train, y_test = train_test_split(values, to_predict, random_state=1)

classifier = RandomForestClassifier(n_jobs=10)
predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
scores = calculate_scores(y_test, predicted)
