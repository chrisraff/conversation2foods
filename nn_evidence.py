import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 70)
        self.fc2 = nn.Linear(70, 60)
        self.fc3 = nn.Linear(60, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X
    
if __name__ == "__main__":
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
    X_train = Variable(torch.Tensor(X_train).float())
    X_test = Variable(torch.Tensor(X_test).float())
    y_train = Variable(torch.Tensor(y_train).long())
    y_test = Variable(torch.Tensor(y_test).long())

    net = Net()

    criterion = nn.CrossEntropyLoss()# cross entropy loss

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(10000):
        optimizer.zero_grad()
        out = net(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print('number of epoch', epoch, 'loss', loss.data.item())
    
    net.eval()

    # predict_out = net(X_test)
    # _, y_pred = torch.max(predict_out, 1)

    # print('prediction accuracy', accuracy_score(y_test.data, y_pred.data))

    # print('macro precision', precision_score(y_test.data, y_pred.data, average='macro'))
    # print('micro precision', precision_score(y_test.data, y_pred.data, average='micro'))
    # print('macro recall', recall_score(y_test.data, y_pred.data, average='macro'))
    # print('micro recall', recall_score(y_test.data, y_pred.data, average='micro'))

    target_names = ['No Evidence', 'Evidence']

    predict_out = net(X_train)
    _, y_pred_train = torch.max(predict_out, 1)
    print("TESTING AGAINST TRAINING DATA")
    print(classification_report(y_train, y_pred_train, target_names=target_names))

    predict_out = net(X_test)
    _, y_pred_test = torch.max(predict_out, 1)
    print("TESTING AGAINST TEST DATA")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    torch.save(net.state_dict(), 'evidencenet.nn')
