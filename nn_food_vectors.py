import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import coverage_error # classification_report

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


# load the foods
with open('all_foods.txt', 'r') as f:
    all_foods = f.readlines()

food_to_idx = {food.strip(): i for i, food in enumerate(all_foods)}


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 100)
        self.fc3 = nn.Linear(100, len(all_foods))
        # self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()


    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc3(X))
        # X = self.softmax(X)
        X = self.sig(X)

        return X
    
if __name__ == "__main__":
    # load dataset
    df = pd.read_csv('bert_food_vectors_data.csv', index_col=0)
    X=df.drop(all_foods,axis=1)
    Y=df[all_foods]

    # X, Y = balanced_subsample(X.values, Y.values)

    print(X.shape, Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X.values,Y.values,random_state=0,test_size=0.3)

    # wrap up with Variable in pytorch
    X_train = Variable(torch.Tensor(X_train).float())
    X_test = Variable(torch.Tensor(X_test).float())
    y_train = Variable(torch.Tensor(y_train).float())
    y_test = Variable(torch.Tensor(y_test).float())


    net = Net()

    criterion = nn.MSELoss()# cross entropy loss

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(101):
        optimizer.zero_grad()
        out = net(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print('number of epoch', epoch, 'loss', loss.data.item())

    predict_out = net(X_test)
    # print(predict_out)
    y_pred = predict_out >= 0.9
    # _, y_pred = torch.max(predict_out, 1)

    print(y_pred)
    print(y_pred.sum())

    # print('prediction accuracy', accuracy_score(y_test.data, y_pred.data))

    # print('macro precision', precision_score(y_test.data, y_pred.data, average='macro'))
    # print('micro precision', precision_score(y_test.data, y_pred.data, average='micro'))
    # print('macro recall', recall_score(y_test.data, y_pred.data, average='macro'))
    # print('micro recall', recall_score(y_test.data, y_pred.data, average='micro'))

    target_names = all_foods
    # print(classification_report(y_test.data, predict_out.data, target_names=target_names))
    print(coverage_error(y_test.data, predict_out.data))

    torch.save(net.state_dict(), 'evidencenet.nn')
