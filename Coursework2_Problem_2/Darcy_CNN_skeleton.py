import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py

# Create results folder if it does not exist
result_folder = 'Coursework2_Problem_2/results'
os.makedirs(result_folder, exist_ok=True)

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
        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
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
        self.data = h5py.File(self.file_path, 'r')

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)
    
# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return (x * (self.std + self.eps)) + self.mean

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self, channel_width=64):
        super(CNN, self).__init__()
        # A simple fully convolutional network that preserves spatial dimensions
        self.layers = nn.Sequential(
            nn.Conv2d(1, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: (batch_size, H, W); add channel dimension for CNN
        x = x.unsqueeze(1)
        out = self.layers(x)
        # Remove the channel dimension before returning
        out = out.squeeze(1)
        return out

if __name__ == '__main__':
    ############################# Data Processing #############################
    # Read data from .mat files
    train_path = 'Coursework2_Problem_2/Darcy_2D_data_train.mat'
    test_path = 'Coursework2_Problem_2/Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize the data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)
    u_train = u_normalizer.encode(u_train)
    u_test = u_normalizer.encode(u_test)

    print("a_train:", a_train.shape)
    print("u_train:", u_train.shape)
    print("a_test:", a_test.shape)
    print("u_test:", u_test.shape)

    # Create DataLoader for training data
    batch_size = 20
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    ############################# Define and Train Network #############################
    # Create RNN instance, define loss function and optimizer
    channel_width = 64
    net = CNN(channel_width=channel_width)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    # Define loss function, optimizer, and learning rate scheduler
    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    epochs = 20  # Number of epochs
    print("Start training CNN for {} epochs...".format(epochs))
    start_time = time()
    
    loss_train_list = []
    loss_test_list = []
    epochs_list = []
    
    for epoch in range(epochs):
        net.train()
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
    
        # Evaluate on test data
        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test).item()

        avg_train_loss = trainloss / len(train_loader)
        loss_train_list.append(avg_train_loss)
        loss_test_list.append(testloss)
        epochs_list.append(epoch)

        if epoch % 10 == 0:
            print("Epoch: {}, Train Loss: {:.5f}, Test Loss: {:.5f}".format(epoch, avg_train_loss, testloss))

    total_time = time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time: {}'.format(total_time_str))
    print("Final Train Loss: {:.5f}".format(loss_train_list[-1]))
    print("Final Test Loss: {:.5f}".format(loss_test_list[-1]))
    
    ############################# Plot Loss Curves #############################
    plt.figure(figsize=(8,5))
    plt.plot(epochs_list, loss_train_list, label='Train Loss')
    plt.plot(epochs_list, loss_test_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, max(max(loss_train_list), max(loss_test_list))*1.1)
    plt.legend()
    plt.grid(True)
    plt.title('Training and Test Loss vs. Epochs')
    plt.savefig(os.path.join(result_folder, 'loss_plot.png'))
    plt.close()
    
    ############################# Contour Plots for a Single Test Sample #############################
    sample_index = 0
    sample_a = a_test[sample_index:sample_index+1]  # shape: (1, H, W)
    sample_true = u_test[sample_index].numpy()       # shape: (H, W)
    
    net.eval()
    with torch.no_grad():
        sample_pred = net(sample_a)
        sample_pred = u_normalizer.decode(sample_pred)
    sample_pred = sample_pred.squeeze().numpy()

    # Create grid for contour plotting (assuming square domain)
    nx, ny = sample_true.shape
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    plt.figure(figsize=(12,5))
    
    # True solution contour
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(X, Y, sample_true, levels=20, cmap='viridis')
    plt.colorbar(cp1)
    plt.title('True Solution $u(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Predicted solution contour
    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(X, Y, sample_pred, levels=20, cmap='viridis')
    plt.colorbar(cp2)
    plt.title('Predicted Solution $u(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'contour_plot.png'))
    plt.close()