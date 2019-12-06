import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


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
    df = pd.read_csv('bert_data_augmented.csv', index_col=0)
    print('done')
    X=df.drop(['labels'],axis=1)
    Y=df['labels']

    # X, Y = balanced_subsample(X.values, Y.values)

    # print(X.shape, Y.shape)

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

    net = Net()

    criterion = nn.CrossEntropyLoss()# cross entropy loss

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(10000):
        optimizer.zero_grad()
        out = net(X_train_balanced)
        loss = criterion(out, y_train_balanced)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print('number of epoch', epoch, 'loss', loss.data.item())

    # predict_out = net(X_test)
    # _, y_pred = torch.max(predict_out, 1)

    # print('prediction accuracy', accuracy_score(y_test.data, y_pred.data))

    # print('macro precision', precision_score(y_test.data, y_pred.data, average='macro'))
    # print('micro precision', precision_score(y_test.data, y_pred.data, average='micro'))
    # print('macro recall', recall_score(y_test.data, y_pred.data, average='macro'))
    # print('micro recall', recall_score(y_test.data, y_pred.data, average='micro'))

    target_names = ['No Evidence', 'Evidence']

    predict_out = net(X_train_balanced)
    _, y_pred_train = torch.max(predict_out, 1)
    print("TESTING AGAINST BALANCED TRAINING DATA")
    print(classification_report(y_train_balanced, y_pred_train, target_names=target_names))

    predict_out = net(X_test_balanced)
    _, y_pred_balanced = torch.max(predict_out, 1)
    print("TESTING AGAINST BALANCED TEST DATA")
    print(classification_report(y_test_balanced, y_pred_balanced, target_names=target_names))

    predict_out = net(X_test_unbalanced)
    _, y_pred_unbalanced = torch.max(predict_out, 1)
    print("TESTING AGAINST UNBALANCED TEST DATA")
    print(classification_report(y_test_unbalanced, y_pred_unbalanced, target_names=target_names))


    torch.save(net.state_dict(), 'evidencenet.nn')
