__author__ = 'Dimas'

import pandas
import re
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif

family_id_mapping = {}


def add_new_features(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"]

    data["NameLength"] = data["Name"].apply(lambda x: len(x))

    add_title_feature(data)

    add_family_id_feature(data)

def add_family_id_feature(data):
    family_ids = data.apply(get_family_id, axis=1)

    family_ids[data["FamilySize"] < 3] = -1

    data["FamilyId"] = family_ids

def get_family_id(row):
    last_name = row["Name"].split(",")[0]

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)
    return ""

def add_title_feature(data):
    titles = data["Name"].apply(get_title)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 9}
    for k, v in title_mapping.items():
        titles[titles == k] = v

    data["Title"] = titles

def fill_missing_data(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

def map_features(data):
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

def predict_with_logistic_regression(data, predictors, target, test_data):
    alg = LogisticRegression(random_state=1)

    alg.fit(data[predictors], data[target])

    scores = cross_validation.cross_val_score(alg, data[predictors], data[target], cv=3)
    # Take the mean of the scores (because we have one for each fold)
    print("Logistic regression score: " + str(scores.mean()))

    return alg.predict(test_data[predictors])

def predict_with_random_forest(data, predictors, target, test_data):
    alg = RandomForestClassifier(random_state=1, n_estimators=1500, min_samples_split=8, min_samples_leaf=2)

    alg.fit(data[predictors], data[target])
    scores = cross_validation.cross_val_score(alg, data[predictors], data[target], cv=3)

    print("Random Forest score: " + str(scores.mean()))

    return alg.predict(test_data[predictors])

def feature_selection(data, predictors, target):

    selector = SelectKBest(f_classif, k=5)
    selector.fit(data[predictors], data[target])

    scores = -np.log10(selector.pvalues_)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()


def main():
    train_data = pandas.read_csv('train.csv')
    test_data = pandas.read_csv('test.csv')

    fill_missing_data(train_data)
    fill_missing_data(test_data)

    map_features(train_data)
    map_features(test_data)

    add_new_features(train_data)
    add_new_features(test_data)

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength",
                  "Title", "FamilyId"]

    #predictors = ["Pclass", "Sex", "Fare", "Title", "Fare"]

    #logistic_predictions = predict_with_logistic_regression(train_data, predictors, "Survived", test_data)
    random_forest_predictions = predict_with_random_forest(train_data, predictors, "Survived", test_data)

    #feature_selection(train_data, predictors, "Survived")


    submission = pandas.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": random_forest_predictions
    })

    submission.to_csv("kaggle.csv", index=False)


if __name__ == '__main__':
    main()
