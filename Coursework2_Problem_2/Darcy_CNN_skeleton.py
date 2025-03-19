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
    "arch": "CNN",
    "epochs": 500,
    "batch_size": 20,
    "learning_rate": 3e-4,
    "channel_width": 128,
    "hidden_layer": 3,
    "scheduler_step": 5000,
    "scheduler_gamma": 0.6,
    "do_train": True
})
config = wandb.config

# Create results folder if it does not exist
result_folder = 'Coursework2_Problem_2/CNN_results'
os.makedirs(result_folder, exist_ok=True)

# Set device to cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self, channel_width=64, hidden_layers=5):
        super(CNN, self).__init__()
        layers = []
        # Input layer: from 1 channel to channel_width channels.
        layers.append(nn.Conv2d(1, channel_width, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        # Add hidden layers.
        for _ in range(hidden_layers):
            layers.append(nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        # Output layer: map channel_width channels back to 1 channel.
        layers.append(nn.Conv2d(channel_width, 1, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, H, W); add channel dimension for CNN
        x = x.unsqueeze(1)
        out = self.layers(x)
        # Remove the channel dimension before returning
        return out.squeeze(1)

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
    net = CNN(config.channel_width,config.hidden_layer).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)
    
    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    
    epochs = config.epochs

    model_weights_path = os.path.join(result_folder, 'cnn_model.pth')
    if config.do_train:
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

        # Save model weights
        torch.save(net.state_dict(), model_weights_path)
        print(f"Model weights saved to {model_weights_path}")

        ############################# Save Loss Curves #############################
        plt.figure(figsize=(6,5), dpi=300)
        plt.plot(epoch_list, loss_train_list, label='Train Loss')
        plt.plot(epoch_list, loss_test_list, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training and Test Loss vs. Epochs')
        plt.ylim(0, max(max(loss_train_list), max(loss_test_list))*1.1)
        loss_plot_path = os.path.join(result_folder, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        wandb.log({"loss_plot": wandb.Image(loss_plot_path)})
        
    else:
        net.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Model weights loaded from {model_weights_path}")
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

    plt.figure(figsize=(12,4), dpi=300)

    # True
    plt.subplot(1,3,1)
    cp1 = plt.contourf(X, Y, sample_true, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cp1)
    plt.title('True Solution $u(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal') 

    # Pred
    plt.subplot(1,3,2)
    cp2 = plt.contourf(X, Y, sample_pred, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cp2)
    plt.title('Predicted Solution $u(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal') 

    # Diff
    plt.subplot(1,3,3)
    cp3 = plt.imshow(abs_diff, cmap='viridis', extent=[0,1,0,1], origin='lower')
    plt.colorbar(cp3)
    plt.title('Absolute Difference')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal') 

    plt.tight_layout()
    contour_plot_path = os.path.join(result_folder, 'contour_plot.png')
    plt.savefig(contour_plot_path)
    plt.close()
    wandb.log({"contour_plot": wandb.Image(contour_plot_path)})