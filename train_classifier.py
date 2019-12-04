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



def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


df = pd.read_csv('bert_data.csv', index_col=0)
X=df.drop(['labels'],axis=1)
Y=df['labels']

X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(X, Y, random_state=0, test_size=0.2, stratify=Y)

balanced_X, balanced_Y = balanced_subsample(X_train_unbalanced.values, y_train_unbalanced.values)

X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(balanced_X, balanced_Y, random_state=0, test_size=0.2, stratify=balanced_Y)

# clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
# clf = LogisticRegression()
clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=5,
    class_weight='balanced'
    )

clf.fit(X_train_balanced, y_train_balanced)
target_names = ['No Evidence', 'Evidence']

y_pred_train = clf.predict(X_train_balanced)
print("TESTING AGAINST BALANCED TRAINING DATA")
print(classification_report(y_train_balanced, y_pred_train, target_names=target_names))

y_pred_balanced = clf.predict(X_test_balanced)
print("TESTING AGAINST BALANCED TEST DATA")
print(classification_report(y_test_balanced, y_pred_balanced, target_names=target_names))

y_pred_unbalanced = clf.predict(X_test_unbalanced)
print("TESTING AGAINST UNBALANCED TEST DATA")
print(classification_report(y_test_unbalanced, y_pred_unbalanced, target_names=target_names))

with open('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
