"""
train a binary classifier on if a bert vector contains
evidence of a food
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report


df = pd.read_csv('bert_data.csv')
X=df.drop(['labels'],axis=1)
Y=df['labels']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, stratify=Y)

# clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
# clf = LogisticRegression()
clf = RandomForestClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=10,
    class_weight='balanced'
    )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

target_names = ['No Evidence', 'Evidence']
print(classification_report(y_test, y_pred, target_names=target_names))