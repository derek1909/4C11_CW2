import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py

# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # print('x.shape',x.shape)
        # print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)
    
# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

# Define network  
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers =
        # construct your own CNN here #

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.squeeze(1)
        return out

if __name__ == '__main__':
    ############################# Data processing #############################
    # Read data from mat
    train_path = 'Darcy_2D_data_train.mat'
    test_path = 'Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)

    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    # Create data loader
    batch_size = 20
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    ############################# Define and train network #############################
    # Create RNN instance, define loss function and optimizer
    channel_width = 64
    net =
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    loss_func = LpLoss()
    optimizer =
    scheduler =

    # Train network
    epochs = 200 # Number of epochs
    print("Start training CNN for {} epochs...".format(epochs))
    start_time = time()
    
    loss_train_list = []
    loss_test_list = []
    x = []
    for epoch in range(epochs):
        net.train(True)
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            output = net(input) # Forward
            output = u_normalizer.decode(output)
            l = loss_func(output, target) # Calculate loss

            optimizer.zero_grad() # Clear gradients
            l.backward() # Backward
            optimizer.step() # Update parameters
            scheduler.step() # Update learning rate

            trainloss += l.item()    

        # Test
        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test).item()

        # Print train loss every 10 epochs
        if epoch % 10 == 0:
            print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss))

        loss_train_list.append(trainloss/len(train_loader))
        loss_test_list.append(testloss)
        x.append(epoch)

    total_time = time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Traing time: {}'.format(total_time_str))
    print("Train loss:{}".format(trainloss/len(train_loader)))
    print("Test loss:{}".format(testloss))
    
    ############################# Plot #############################
    plt.figure(1)
    plt.plot(x, loss_train_list, label='Train loss')
    plt.plot(x, loss_test_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.legend()
    plt.grid()
