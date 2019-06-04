import numpy as np
import torch  # 基本モジュール
from torch.autograd import Variable  # 自動微分用
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連
import torchvision  # 画像関連
from torchvision import datasets, models, transforms  # 画像用データセット諸々

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.input(input)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.output(hidden)
        output = F.log_softmax(output)
        return output


class StackedGRU(nn.Module):
    def __init__(self, length, input_size, gru_input_size, hidden_size, output_size):
        self.input_size = input_size
        self.gru_input_size = gru_input_size
        self.length = length
        super(StackedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(gru_input_size, hidden_size)
        self.linear = nn.Linear(hidden_size+input_size, output_size)

    def forward(self, input):
        hidden, _ = self.gru(input[:, :-self.input_size].view(-1, self.length, 600))
        output = self.linear(torch.cat((hidden[:, -1], input[:, -self.input_size:]), dim=1))
        output = F.log_softmax(output)
        return output

    def initHidden(self):
        return [torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)]


def data_loader(X_train, y_train, X_test, y_test):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train).long())
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test).long())
    test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)
    return train_loader, test_loader


def build_train_process(model, train_loader, lr=0.001):
    # Loss関数の指定
    criterion = nn.CrossEntropyLoss()

    # Optimizerの指定
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def batch_train(epoch):
        # データ全てのトータルロス
        running_loss = 0.0
        correct = 0
        total = 0
        for num, data in enumerate(train_loader):
            # 入力データ・ラベルに分割
            # get the inputs
            inputs, labels = data

            # Variableに変形
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # optimizerの初期化
            # zero the parameter gradients
            optimizer.zero_grad()

            # 一連の流れ
            # forward + backward + optimize
            outputs = model(inputs)

            # ここでラベルデータに対するCross-Entropyがとられる
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # count train-accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # ロスの表示
            # print statistics
            running_loss += loss.data

        print('[%d, %5d] train accuracy: %.3f' % (epoch + 1, total, correct / total))
        print('[%d, %5d] train loss: %.3f' % (epoch + 1, total, running_loss))
        running_loss = 0.0
        return 100 * correct / total, running_loss

    return batch_train


def build_test_process(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    def test(time=0, epoch=0, sentences=None):
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels)
        if epoch % 300 == 0:
            with open('log_{}.txt'.format(time), 'w') as l:
                l.write('the number of errors: {}\n'.format(int(sum(labels==predicted))))
                l.write('\n'.join([s for ans, pred, s in zip(labels, predicted, sentences) if not ans == pred]))
        print('Accuracy of the network on the %5d test images: %5f %%' % (total, 100 * correct / total))
        return 100 * correct / total, test_loss

    return test


def build_categories_test_process(model, test_loader, categories=None):
    def categories_test():
        class_correct = [0.] * len(categories)
        class_total = [0.] * len(categories)
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(data[0])):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(len(categories)):
            print('Accuracy of label %1d %5s : %2d %%' % (i, categories[i], 100 * class_correct[i] / class_total[i] if not class_total[i] == 0 else -1))
        return [100 * class_correct[i] / class_total[i] if not class_total[i] == 0 else -1 for i in range(len(categories))]
    return categories_test
