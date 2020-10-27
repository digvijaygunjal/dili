import pandas as pd
import xgboost as xgb

from src.rfc import classify_and_predict, calculate_scores

if __name__ == "__main__":
    train = pd.read_csv('./data/intermediate/ncrt_train_maccs_morgan_fingerprints.csv', index_col=0).reindex()
    test = pd.read_csv('./data/intermediate/ncrt_test_maccs_morgan_fingerprints.csv', index_col=0).reindex()

    x_train = train.iloc[:, :-1]
    x_test = test.iloc[:, :-1]
    y_train = train['label']
    y_test = test['label']

    classifier = xgb.XGBClassifier()
    predicted, model = classify_and_predict(x_train, x_test, y_train, classifier)
    scores = calculate_scores(y_test, predicted)
    print(scores)
