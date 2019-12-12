'''
use the neural network from nn.py to generate the statistics
without needing to retrain the model
'''
import numpy as np
import pandas as pd
import torch
from nn_evidence import Net
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == "__main__":
    
    cuda = torch.device('cuda')

    # load neural network
    model = Net()
    model.load_state_dict(torch.load('evidencenet.nn'))
    model = model.to(cuda)
    model.eval()

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

    # wrap up with Variable in pytorch
    X_train = Variable(torch.Tensor(X_train).float()).to(cuda)
    X_test = Variable(torch.Tensor(X_test).float()).to(cuda)
    y_train = Variable(torch.Tensor(y_train).long()).to(cuda)
    y_test = Variable(torch.Tensor(y_test).long()).to(cuda)

    #----------------------------------------------------------------
    # evaluate model

    target_names = ['No Evidence', 'Evidence']

    predict_out = model(X_train)
    _, y_pred_train = torch.max(predict_out, 1)
    print("TESTING AGAINST TRAINING DATA")
    print(classification_report(y_train.cpu(), y_pred_train.cpu(), target_names=target_names))

    predict_out = model(X_test)
    _, y_pred_testing = torch.max(predict_out, 1)
    print("TESTING AGAINST TEST DATA")
    print(classification_report(y_test.cpu(), y_pred_testing.cpu(), target_names=target_names))
