import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Initialize wandb
wandb.init(project="Darcy_Flow", config={
    "arch": "FNO",
    "epochs": 500,
    "batch_size": 50,
    "learning_rate": 1e-4,
    "modes": 6,
    "width": 16,
    "scheduler_step": 200,
    "scheduler_gamma": 0.6
})
config = wandb.config

# Create results folder if it does not exist
result_folder = 'Coursework2_Problem_2/FNO_results'
os.makedirs(result_folder, exist_ok=True)

# Set device to cuda if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms / y_norms) if self.size_average else torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
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

# Define spectral convolution layer (Fourier layer)
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# Define a simple MLP (applied as a 1x1 convolution)
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

# Define the Fourier Neural Operator (FNO) architecture
class FNO(nn.Module):
    def __init__(self, modes1, modes2, width):
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

        # Lift the input (which has 3 channels: a(x,y), x, y) to the desired width.
        self.p = nn.Linear(3, self.width)
        
        # Four Fourier layers: each with a spectral convolution and a local (1x1) convolution.
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()
        
        # Project from width to output channel (1: u(x,y))
        self.q = MLP(self.width, 1, self.width * 4)

    def forward(self, x):
        grid = self.get_grid(x.shape).to(x.device)
        # Concatenate input a(x,y) with spatial grid (x,y)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)  # shape: (batch, x, y, 3)
        x = self.p(x)  # lift to width channels; shape: (batch, x, y, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, x, y)
        
        # Layer 1
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        # Layer 2
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        # Layer 3
        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        # Layer 4
        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)
        
        x = self.q(x)
        x = x.squeeze(1)
        return x
    
    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

if __name__ == '__main__':
    ############################# Data processing #############################
    # Read data from .mat files
    train_path = 'Coursework2_Problem_2/Darcy_2D_data_train.mat'
    test_path = 'Coursework2_Problem_2/Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Move raw data to device first
    a_train, u_train = a_train.to(device), u_train.to(device)
    a_test, u_test = a_test.to(device), u_test.to(device)

    # Normalize the data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)

    # Create data loader
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    
    ############################# Define and train network #############################
    modes = config.modes
    width = config.width
    net = FNO(modes, modes, width).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)
    
    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    
    epochs = config.epochs
    
    loss_train_list = []
    loss_test_list = []
    epoch_list = []
    
    # Training loop with tqdm
    with tqdm(total=epochs, initial=0, desc="Training", unit="epoch", dynamic_ncols=True) as pbar_epoch:
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            for input, target in train_loader:
                input, target = input.to(device), target.to(device)
                output = net(input) # Forward
                output = u_normalizer.decode(output)
                l = loss_func(output, target) # Calculate loss

                optimizer.zero_grad() # Clear gradients
                l.backward() # Backward
                optimizer.step() # Update parameters
                scheduler.step() # Update learning rate

                train_loss += l.item()
            # Evaluation
            net.eval()
            with torch.no_grad():
                test_output = net(a_test.to(device))
                test_output = u_normalizer.decode(test_output)
                test_loss = loss_func(test_output, u_test.to(device)).item()
            avg_train_loss = train_loss / len(train_loader)
            loss_train_list.append(avg_train_loss)
            loss_test_list.append(test_loss)
            epoch_list.append(epoch)
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "test_loss": test_loss})

            pbar_epoch.update(1)
            pbar_epoch.set_postfix({"Train Loss": f"{avg_train_loss:.3e}", "Test Loss": f"{test_loss:.3e}"})
    
    print("Final Train Loss: {:.5f}".format(loss_train_list[-1]))
    print("Final Test Loss: {:.5f}".format(loss_test_list[-1]))
    
    ############################# Save Loss Curves #############################
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, loss_train_list, label='Train Loss')
    plt.plot(epoch_list, loss_test_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, max(max(loss_train_list), max(loss_test_list))*1.1)
    plt.legend()
    plt.grid(True)
    plt.title('Training and Test Loss vs. Epochs')
    loss_plot_path = os.path.join(result_folder, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    wandb.log({"loss_plot": wandb.Image(loss_plot_path)})
    
    ############################# Contour Plots for a Single Test Sample #############################
    sample_index = 0
    sample_a = a_test[sample_index:sample_index+1]  # shape: (1, H, W)
    sample_true = u_test[sample_index].cpu().numpy()   # shape: (H, W)
    
    net.eval()
    with torch.no_grad():
        sample_pred = net(sample_a.to(device))
        sample_pred = u_normalizer.decode(sample_pred)
    sample_pred = sample_pred.squeeze().cpu().numpy()
    
    nx, ny = sample_true.shape
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Use common color limits
    vmin = min(sample_true.min(), sample_pred.min())
    vmax = max(sample_true.max(), sample_pred.max())

    # Abs Diff
    abs_diff = np.abs(sample_true - sample_pred)

    plt.figure(figsize=(18,5))

    # True
    plt.subplot(1,3,1)
    cp1 = plt.contourf(X, Y, sample_true, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cp1)
    plt.title('True Solution $u(x)$')
    plt.xlabel('x')
    plt.ylabel('y')

    # Pred
    plt.subplot(1,3,2)
    cp2 = plt.contourf(X, Y, sample_pred, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cp2)
    plt.title('Predicted Solution $u(x)$')
    plt.xlabel('x')
    plt.ylabel('y')

    # Diff
    plt.subplot(1,3,3)
    cp3 = plt.imshow(abs_diff, cmap='viridis', extent=[0,1,0,1], origin='lower')
    plt.colorbar(cp3)
    plt.title('Absolute Difference')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    contour_plot_path = os.path.join(result_folder, 'contour_plot.png')
    plt.savefig(contour_plot_path)
    plt.close()
    wandb.log({"contour_plot": wandb.Image(contour_plot_path)})