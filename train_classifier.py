"""
train a binary classifier on if a bert vector contains
evidence of a food
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



print('loading dataset')
df_train = pd.read_csv('bert_evidence_train.csv', index_col=0)
X_train = df_train.drop(['labels'],axis=1)
y_train = df_train['labels']

df_test = pd.read_csv('bert_evidence_test.csv', index_col=0)
X_test = df_test.drop(['labels'],axis=1)
y_test = df_test['labels']

print('done')

X_train, y_train = X_train.values, y_train.values
X_test, y_test = X_test.values, y_test.values

# clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
clf = LogisticRegression()
# clf = RandomForestClassifier( # takes forever with large set
#     n_estimators=1000,
#     max_depth=7,
#     min_samples_split=5,
#     min_samples_leaf=5,
#     class_weight='balanced'
#     )

clf.fit(X_train, y_train)
target_names = ['No Evidence', 'Evidence']

y_pred_train = clf.predict(X_train)
print("TESTING AGAINST TRAINING DATA")
print(classification_report(y_train, y_pred_train, target_names=target_names))

y_pred_test = clf.predict(X_test)
print("TESTING AGAINST TEST DATA")
print(classification_report(y_test, y_pred_test, target_names=target_names))

with open('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
