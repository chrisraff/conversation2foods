'''
use the neural network from nn.py to generate the statistics
without needing to retrain the model
'''
import numpy as np
import pandas as pd
import torch
from nn_evidence import Net
from torch.autograd import Variable
from bertinator import get_bert_vector
from nn_evidence import balanced_subsample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # load neural network
    model = Net()
    model.load_state_dict(torch.load('evidencenet.nn'))
    model.eval()

    # load dataset
    print('loading dataset')
    df = pd.read_csv('bert_data_augmented.csv', index_col=0)
    print('done')
    X=df.drop(['labels'],axis=1)
    Y=df['labels']

    X, Y = X.values, Y.values

    X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(X, Y, random_state=0, test_size=0.2, stratify=Y)

    balanced_X, balanced_Y = balanced_subsample(X_train_unbalanced, y_train_unbalanced)

    X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(balanced_X, balanced_Y, random_state=0, test_size=0.2, stratify=balanced_Y)

    # wrap up with Variable in pytorch
    X_train_balanced = Variable(torch.Tensor(X_train_balanced).float())
    X_test_balanced = Variable(torch.Tensor(X_test_balanced).float())
    y_train_balanced = Variable(torch.Tensor(y_train_balanced).long())
    y_test_balanced = Variable(torch.Tensor(y_test_balanced).long())

    X_train_unbalanced = Variable(torch.Tensor(X_train_unbalanced).float())
    X_test_unbalanced = Variable(torch.Tensor(X_test_unbalanced).float())
    y_train_unbalanced = Variable(torch.Tensor(y_train_unbalanced).long())
    y_test_unbalanced = Variable(torch.Tensor(y_test_unbalanced).long())
    
    print('loading unaugmented  dataset')
    df_u = pd.read_csv('bert_data.csv', index_col=0)
    print('done')
    X_unaugmented=df_u.drop(['labels'],axis=1)
    Y_unaugmented=df_u['labels']

    # X, Y = balanced_subsample(X.values, Y.values)

    # print(X.shape, Y.shape)

    X_unaugmented, Y_unaugmented = X_unaugmented.values, Y_unaugmented.values

    X_train_unbalanced_unaugmented, X_test_unbalanced_unaugmented, y_train_unbalanced_unaugmented, y_test_unbalanced_unaugmented = train_test_split(X_unaugmented, Y_unaugmented, random_state=0, test_size=0.2, stratify=Y_unaugmented)

    # X_train_balanced_unaugmented, X_test_balanced_unaugmented, y_train_balanced_unaugmented, y_test_balanced_unaugmented = train_test_split(balanced_X_unaugmented, balanced_Y_unaugmented, random_state=0, test_size=0.2, stratify=balanced_Y_unaugmented)

    # wrap up with Variable in pytorch

    X_train_unbalanced_unaugmented = Variable(torch.Tensor(X_train_unbalanced_unaugmented).float())
    X_test_unbalanced_unaugmented = Variable(torch.Tensor(X_test_unbalanced_unaugmented).float())
    y_train_unbalanced_unaugmented = Variable(torch.Tensor(y_train_unbalanced_unaugmented).long())
    y_test_unbalanced_unaugmented = Variable(torch.Tensor(y_test_unbalanced_unaugmented).long())

    #----------------------------------------------------------------
    # evaluate model
    target_names = ['No Evidence', 'Evidence']

    predict_out = model(X_train_balanced)
    _, y_pred_train = torch.max(predict_out, 1)
    print("TESTING AGAINST BALANCED TRAINING DATA")
    print(classification_report(y_train_balanced, y_pred_train, target_names=target_names))

    predict_out = model(X_test_balanced)
    _, y_pred_balanced = torch.max(predict_out, 1)
    print("TESTING AGAINST BALANCED TEST DATA")
    print(classification_report(y_test_balanced, y_pred_balanced, target_names=target_names))

    predict_out = model(X_test_unbalanced)
    _, y_pred_unbalanced = torch.max(predict_out, 1)
    print("TESTING AGAINST UNBALANCED TEST DATA")
    print(classification_report(y_test_unbalanced, y_pred_unbalanced, target_names=target_names))

    predict_out = model(X_test_unbalanced_unaugmented)
    _, y_pred_unbalanced_unaugmented = torch.max(predict_out, 1)
    print("TESTING AGAINST UNBALANCED UNAUGMENTED TEST DATA")
    print(classification_report(y_test_unbalanced_unaugmented, y_pred_unbalanced_unaugmented, target_names=target_names))
