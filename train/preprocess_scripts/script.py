# -*- coding: utf-8 -*-

# import required packages
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# from matplotlib import pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle
import os
import joblib

# Read train data
train_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../attacker_data/train_attacker_2022.csv"), index_col="id", skipinitialspace=True
)

# Clean data

# remove whitespace and dash in columns and values
train_data.columns = train_data.columns.str.replace(" ", "")

for (columnName, columnData) in train_data.iteritems():
    train_data[columnName] = columnData.replace(["-   ", " -   ", " "], "")


# drop any columns with null values > 60%
columns_to_drop = [
    col
    for col in train_data.columns
    if (train_data[col].isna().sum() / len(train_data[col]) > 0.6)
]
train_data = train_data.drop(columns=columns_to_drop, axis=1)

# get list of categorical features
list_categorical_cols = list(train_data.columns[train_data.dtypes == "O"])

categorical_cols_exceptions = ["value", "review_value"]

# remove any categorical column with more than 4 features
for cat_feature in list_categorical_cols:
    if (
        train_data[cat_feature].value_counts().shape[0] > 4
        and cat_feature not in categorical_cols_exceptions
    ):
        train_data = train_data.drop(cat_feature, axis=1)

# transform value and review_value to numeric
train_data["value"] = pd.to_numeric(train_data["value"].str.replace(",", ""))
train_data["review_value"] = pd.to_numeric(
    train_data["review_value"].str.replace(",", "")
)

# add values to rows with duplicate ids
new_train = train_data.groupby(train_data.index).first()

cols = list(train_data.columns)
train_data.loc[train_data.index.isin(new_train.index), cols] = new_train[cols]

# Fill missing data and transform data
# init imputer fn
def imputer(df):
    mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    for col in df.columns:
        if df[col].isnull().any() == True:
            if df[col].dtypes == "O":
                df[col] = df[col].apply(lambda x: col + "_No" if x is None else x)
            else:
                df[col] = mode.fit_transform(df[[col]])
    return df


# init confusion matrix visualization
# def confusion_matrix_df(y_test, y_pred):
#     df = pd.DataFrame(
#         confusion_matrix(y_test, y_pred),
#         columns=["Predict No Distress", "Predict Distress"],
#         index=["Actual No Distress", "Actual Distress"],
#     )
#     return df.style.background_gradient(cmap="Greens")


# impute train data
imputer(train_data)

# one-hot encoding
train_data = pd.get_dummies(train_data)

# save processed data
train_data.to_csv(os.path.join(os.path.dirname(__file__),"../processed/train.csv"))

# Train on processed data

# split dataset
features = ['label','value','num_date_review','dob','unknown_var_1','unknown_var_2','unknown_var_3','unknown_var_7','unknown_var_8','unknown_var_9','unknown_var_10','unknown_var_12','unknown_var_14','unknown_var_15','social_friend_count', 'unknown_var_17']
for feat in train_data.columns:
    if feat not in features:
        train_data = train_data.drop(feat, axis=1)
n_state = 45
X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop("label", axis=1),
    train_data["label"],
    train_size=0.8,
    test_size=0.2,
    random_state=n_state,
)

forest = RandomForestClassifier(random_state=n_state)
model = forest.fit(X_train, y_train)
y_pred_dectree = model.predict(X_test)

# print(classification_report(y_test, y_pred_dectree))
# print(f"Decision Tree Accuracy: {100*round(accuracy_score(y_test, y_pred_dectree),2)}%")
# confusion_matrix_df(y_test, y_pred_dectree)

# save the model
joblib.dump(model, open(os.path.join(os.path.dirname(__file__),"../../model.pkl"), "wb"), 9)
