import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, matthews_corrcoef, roc_auc_score, \
    balanced_accuracy_score

from src.scoring import specificity, sensitivity


def classify_and_predict(x_train, x_test, y_train, model):
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    return y_predicted, model


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


if __name__ == "__main__":
    train = pd.read_csv('./data/intermediate/ncrt_train_maccs_morgan_fingerprints.csv', index_col=0).reindex()
    test = pd.read_csv('./data/intermediate/ncrt_test_maccs_morgan_fingerprints.csv', index_col=0).reindex()

    x_train = train.iloc[:, :-1]
    x_test = test.iloc[:, :-1]
    y_train = train['label']
    y_test = test['label']

    classifier = RandomForestClassifier(n_jobs=10)
    predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
    scores = calculate_scores(y_test, predicted)
    print(scores)
