import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py
import torch.nn.functional as F

# Define Lp loss
class LpLoss(object):


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
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 =
        self.conv2 =
        self.conv3 =
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 =
        self.mlp2 =
        self.mlp3 =
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 =
        self.w2 =
        self.w3 =
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 =
        x1 =
        x2 =
        x =
        x =

        x1 =
        x1 =
        x2 =
        x =
        x =

        x1 =
        x1 =
        x2 =
        x =

        x = self.q(x)
        x = x.squeeze(1)
        return x
    
    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

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
    batch_size =
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    ############################# Define and train network #############################
    # Create RNN instance, define loss function and optimizer
    modes = 12
    width = 32
    net = FNO(modes, modes, width)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.6)

    # Train network
    epochs = 200
    print("Start training FNO for {} epochs...".format(epochs))
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
    plt.legend()
    plt.grid()
