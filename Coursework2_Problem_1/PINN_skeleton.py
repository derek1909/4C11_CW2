import os
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.tri as mtri
from tqdm import tqdm
import yaml

# Initialize wandb project
wandb.init(project="Plate_PINN")

# Log configuration parameters to wandb
config = wandb.config
config.learning_rate = 3e-4
config.measurementloss = True
config.iterations = int(1e5)
config.Disp_layer = [2, 300, 300, 2]
config.Stress_layer = [2, 400, 400, 3]
config.E = 10.0
config.mu = 0.3
config.scheduler_step = 2000
config.scheduler_gamma = 0.75
config.do_train = False

gpu_id = 0 if config.measurementloss else 1
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Define the DenseNet neural network
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x

############################# Data Processing #############################
# Specify the path to your .mat data file
path = './Coursework2_Problem_1/plate_data.mat'
data = scipy.io.loadmat(path)
torch.set_default_tensor_type(torch.DoubleTensor)

# Load and move tensors to CUDA
L_boundary = torch.tensor(data['L_boundary'], dtype=torch.float64).to(device)
R_boundary = torch.tensor(data['R_boundary'], dtype=torch.float64).to(device)
T_boundary = torch.tensor(data['T_boundary'], dtype=torch.float64).to(device)
B_boundary = torch.tensor(data['B_boundary'], dtype=torch.float64).to(device)
C_boundary = torch.tensor(data['C_boundary'], dtype=torch.float64).to(device)
Boundary   = torch.tensor(data['Boundary'], dtype=torch.float64, requires_grad=True).to(device)

# truth solution from FEM
disp_truth = torch.tensor(data['disp_data'], dtype=torch.float64).to(device)

# connectivity matrix - this helps you to plot the figure but we do not need it for PINN
t_connect  = torch.tensor(data['t'].astype(float), dtype=torch.float64).to(device)

# all collocation points
x_full = torch.tensor(data['p_full'], dtype=torch.float64, requires_grad=True).to(device)

# collocation points excluding the boundary
x = torch.tensor(data['p'], dtype=torch.float64, requires_grad=True).to(device)

# This chooses 50 fixed points from the truth solution, which we will use for part (e)
rand_index = torch.randint(0, len(x_full), (50,)).to(device)
disp_fix = disp_truth[rand_index, :]

# We will use two neural networks for the problem:
# NN1: to map the coordinates [x,y] to displacement u
# NN2: to map the coordinates [x,y] to the stresses [sigma_11, sigma_22, sigma_12]
# What we will do later is to first compute strain by differentiate the output of NN1
# And then we compute a augment stress using Hook's law to find an augmented stress sigma_a
# And we will require the output of NN2 to match sigma_a  - we shall do this by adding a term in the loss function
# This will help us to avoid differentiating NN1 twice (why?)
# As it is well known that PINN suffers from higher order derivatives

Disp_layer = config.Disp_layer  # Displacement network layers
Stress_layer = config.Stress_layer  # Stress network layers

stress_net = DenseNet(Stress_layer, nn.Tanh).to(device) # Note we choose hyperbolic tangent as an activation function here
disp_net = DenseNet(Disp_layer, nn.Tanh).to(device)

# Define material properties and move to CUDA
E = config.E        
mu = config.mu        
stiff = E/(1-mu**2)*torch.tensor([[1,mu,0],[mu,1,0],[0,0,(1-mu)/2]]) # Hooke's law for plane stress
stiff = stiff.unsqueeze(0).to(device)

# PINN requires super large number of iterations to converge (on the order of 50e^3-100e^3)
iterations = config.iterations

# Define loss function
loss_func = nn.MSELoss()

# Broadcast stiffness for batch multiplication later
stiff_bc = stiff
stiff_full = stiff
stiff = torch.broadcast_to(stiff, (len(x),3,3))
stiff_bc = torch.broadcast_to(stiff_bc, (len(Boundary),3,3))
stiff_full = torch.broadcast_to(stiff_full, (len(x_full), 3, 3))

params = list(stress_net.parameters()) + list(disp_net.parameters())

# Define optimizer and learning rate scheduler
optimizer = torch.optim.Adam(params, lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

mse_loss_func = nn.MSELoss()

# Create directory for saving results if it doesn't exist
results_folder = f'./Coursework2_Problem_1/results/enhanced_{config.measurementloss}'
os.makedirs(results_folder, exist_ok=True)
yaml_path = os.path.join(results_folder, "training_history.yaml")


# Lists to store loss metrics for plotting
loss_epochs = []
loss_pde = []
loss_constitutive = []
loss_boundary = []
loss_measurement = []
loss_total = []
loss_disp_mse = []

if config.do_train:
    # Training loop
    with tqdm(total=iterations, initial=0, desc="Training", unit="epoch", dynamic_ncols=True) as pbar_epoch:
        for epoch in range(iterations):
            optimizer.zero_grad()

            # To compute stress from stress net
            sigma = stress_net(x)
            # To compute displacement from disp net
            disp = disp_net(x)

            # Separate displacement components
            u = disp[:, 0]
            v = disp[:, 1]

            # displacement in x direction
            u = disp[:,0]
            # displacement in y direction
            v = disp[:,1]

            # find the derivatives
            dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True)[0]
            dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),create_graph=True)[0]

            # Define strain components
            e_11 = dudx[:, 0].unsqueeze(1)
            e_22 = dvdx[:, 1].unsqueeze(1)
            e_12 = 0.5 * (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)
            e = torch.cat((e_11, e_22, e_12), 1).unsqueeze(2)

            # Define augment stress
            sig_aug = torch.bmm(stiff, e).squeeze(2)

            # Define constitutive loss - forcing the augment stress to be equal to the neural network stress
            loss_cons = loss_func(sig_aug, sigma)

            # find displacement and stress at the boundaries
            disp_bc = disp_net(Boundary)
            sigma_bc = stress_net(Boundary)
            u_bc = disp_bc[:,0]
            v_bc = disp_bc[:,1]

            # Compute the strain and stresses at the boundary
            dudx_bc = torch.autograd.grad(u_bc, Boundary, grad_outputs=torch.ones_like(u_bc), create_graph=True)[0]
            dvdx_bc = torch.autograd.grad(v_bc, Boundary, grad_outputs=torch.ones_like(v_bc), create_graph=True)[0]

            e_11_bc = dudx_bc[:,0].unsqueeze(1)
            e_22_bc = dvdx_bc[:,1].unsqueeze(1)
            e_12_bc = 0.5 * (dudx_bc[:, 1] + dvdx_bc[:, 0]).unsqueeze(1)
            e_bc = torch.cat((e_11_bc,e_22_bc,e_12_bc), 1).unsqueeze(2)

            sig_aug_bc = torch.bmm(stiff_bc, e_bc).squeeze(2)

            # force the augment stress to agree with the NN stress at the boundary
            loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

            #============= Equilibrium (PDE residual) ===================#
            sig_11 = sigma[:, 0]
            sig_22 = sigma[:, 1]
            sig_12 = sigma[:, 2]

            dsig11dx = torch.autograd.grad(sig_11, x, grad_outputs=torch.ones_like(sig_11), create_graph=True)[0]
            dsig22dx = torch.autograd.grad(sig_22, x, grad_outputs=torch.ones_like(sig_22), create_graph=True)[0]
            dsig12dx = torch.autograd.grad(sig_12, x, grad_outputs=torch.ones_like(sig_12), create_graph=True)[0]

            eq_x1 = dsig11dx[:, 0] + dsig12dx[:, 1]
            eq_x2 = dsig12dx[:, 0] + dsig22dx[:, 1]

            # zero body forces
            f_x1 = torch.zeros_like(eq_x1)
            f_x2 = torch.zeros_like(eq_x2)

            loss_eq1 = loss_func(eq_x1, f_x1)
            loss_eq2 = loss_func(eq_x2, f_x2)

            #========= Boundary Conditions ========================#
            # Prescribed tractions
            tau_R = 0.1  # Right boundary normal traction
            tau_T = 0.0  # Top boundary traction
            u_L = disp_net(L_boundary)
            u_B = disp_net(B_boundary)
            sig_R = stress_net(R_boundary)
            sig_T = stress_net(T_boundary)
            sig_C = stress_net(C_boundary)

            # Symmetry boundary condition left
            loss_BC_L = loss_func(u_L[:,0], torch.zeros_like(u_L[:,0]))
            # Symmetry boundary condition bottom
            loss_BC_B = loss_func(u_B[:, 1], torch.zeros_like(u_B[:, 1]))
            # Force boundary condition right
            loss_BC_R = loss_func(sig_R[:, 0], tau_R*torch.ones_like(sig_R[:, 0])) \
                        + loss_func(sig_R[:, 2],  torch.zeros_like(sig_R[:, 2]))

            loss_BC_T = + loss_func(sig_T[:, 1], tau_T*torch.ones_like(sig_T[:, 1]))   \
                        + loss_func(sig_T[:, 2],  torch.zeros_like(sig_T[:, 2]))

            # traction free on circle
            loss_BC_C = loss_func(sig_C[:,0]*C_boundary[:,0]+sig_C[:,2]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))  \
                        + loss_func(sig_C[:,2]*C_boundary[:,0]+sig_C[:,1]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))

            loss_fix = torch.tensor(0.0, device=device, requires_grad=True)
            if config.measurementloss:
                x_fix = x_full[rand_index, :]
                u_fix = disp_net(x_fix)
                loss_fix = loss_func(u_fix,disp_fix)

            # Define the total loss function:
            # 1. loss_eq1 + loss_eq2: PDE residual error (equilibrium)
            # 2. loss_cons + loss_cons_bc: Constitutive loss (enforcing Hooke's law consistency)
            # 3. loss_BC_L + loss_BC_B + loss_BC_R + loss_BC_T + loss_BC_C: Boundary condition losses
            # 4. 100*loss_fix: measurement loss
            loss = loss_eq1 + loss_eq2 + loss_cons + loss_cons_bc + \
                    loss_BC_L + loss_BC_B + loss_BC_R + loss_BC_T + loss_BC_C + \
                    100*loss_fix

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 20 == 0:
                u_full_eval = disp_net(x_full)
                mse_eval = mse_loss_func(u_full_eval, disp_truth)
                pde_loss_val = loss_eq1.item() + loss_eq2.item()
                constitutive_loss_val = loss_cons.item() + loss_cons_bc.item()
                boundary_loss_val = (loss_BC_L.item() + loss_BC_B.item() +
                                    loss_BC_R.item() + loss_BC_T.item() + loss_BC_C.item())
                measurement_loss_val = 100 * loss_fix

                wandb.log({
                    "epoch": epoch,
                    "PDE_residual_loss": pde_loss_val,
                    "constitutive_loss": constitutive_loss_val,
                    "boundary_loss": boundary_loss_val,
                    "measurement_loss": measurement_loss_val.item(),
                    "total_loss": loss.item(),
                    "disp_mse": mse_eval.item()
                })

                # Record losses in lists for plotting later
                loss_epochs.append(epoch)
                loss_pde.append(pde_loss_val)
                loss_constitutive.append(constitutive_loss_val)
                loss_boundary.append(boundary_loss_val)
                loss_measurement.append(measurement_loss_val.item())
                loss_total.append(loss.item())
                loss_disp_mse.append(mse_eval.item())

                pbar_epoch.update(20)
                pbar_epoch.set_postfix({"Total Loss": f"{loss.item():.3e}", "Disp MSE": f"{mse_eval.item():.3e}"})

    history = {
        "epoch": loss_epochs,
        "PDE_residual_loss": loss_pde,
        "constitutive_loss": loss_constitutive,
        "boundary_loss": loss_boundary,
        "measurement_loss": loss_measurement,
        "total_loss": loss_total,
        "disp_mse": loss_disp_mse
    }

    with open(yaml_path, "w") as f:
        yaml.dump(history, f)

    print(f"Training history saved to {yaml_path}")

    # Save the model checkpoint after training
    checkpoint_path = os.path.join(results_folder, 'model_checkpoint.pth')
    torch.save({
        'stress_net': stress_net.state_dict(),
        'disp_net': disp_net.state_dict(),
    }, checkpoint_path)
    wandb.save(checkpoint_path)
else:
    # Load the pre-trained model from the checkpoint
    checkpoint_path = os.path.join(results_folder, 'model_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        stress_net.load_state_dict(checkpoint['stress_net'])
        disp_net.load_state_dict(checkpoint['disp_net'])
        stress_net.eval()
        disp_net.eval()
        print("Loaded model from checkpoint.")
    else:
        raise FileNotFoundError("Checkpoint not found. Please train the model first or check the checkpoint path.")

    with open(yaml_path, "r") as f:
        history_loaded = yaml.safe_load(f)

    epochs = history_loaded["epoch"]
    pde_loss = history_loaded["PDE_residual_loss"]
    constitutive_loss = history_loaded["constitutive_loss"]
    boundary_loss = history_loaded["boundary_loss"]
    measurement_loss = history_loaded["measurement_loss"]
    total_loss = history_loaded["total_loss"]
    disp_mse = history_loaded["disp_mse"]  

# ------------------------- EVALUATION AND PLOTTING -------------------------


# After training, create a log-scale plot of all loss values
plt.figure(figsize=(10, 6))
plt.plot(loss_epochs, loss_total, label='Total Loss', linewidth=2)
plt.plot(loss_epochs, loss_pde, label='PDE Residual Loss',linestyle='--')
plt.plot(loss_epochs, loss_constitutive, label='Constitutive Loss',linestyle='--')
plt.plot(loss_epochs, loss_boundary, label='Boundary Loss',linestyle='--')
plt.plot(loss_epochs, loss_measurement, label='Measurement Loss',linestyle='--')
plt.plot(loss_epochs, loss_disp_mse, label='Displacement MSE', linewidth=2)
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Loss (log scale)")
plt.title("Loss History (Log Scale)")
plt.legend()
plt.grid(True)

# Save the plot locally
loss_plot_path = os.path.join(results_folder, 'loss_history_log.png')
plt.savefig(loss_plot_path)
plt.close()
print(f"Loss history plot saved to {loss_plot_path}")

# Log the loss plot image to wandb
wandb.log({"loss_history_log": wandb.Image(loss_plot_path)})


# Evaluate the displacement network over all full domain points
u_full = disp_net(x_full)
u = u_full[:, 0]
v = u_full[:, 1]

# Compute gradients with respect to the full collocation points
dudx = torch.autograd.grad(u, x_full, grad_outputs=torch.ones_like(u), create_graph=True)[0]
dvdx = torch.autograd.grad(v, x_full, grad_outputs=torch.ones_like(v), create_graph=True)[0]

# Compute strain components
e_11 = dudx[:, 0].unsqueeze(1)
e_22 = dvdx[:, 1].unsqueeze(1)
e_12 = 0.5 * (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)
e = torch.cat((e_11, e_22, e_12), 1).unsqueeze(2)

# Compute stresses using the constitutive law (Hooke's law)
sigma_full_calc = torch.bmm(stiff_full, e).squeeze(2)
sigma11 = sigma_full_calc[:, 0].detach().cpu().numpy()
sigma22 = sigma_full_calc[:, 1].detach().cpu().numpy()
sigma12 = sigma_full_calc[:, 2].detach().cpu().numpy()

# For displacement, compute the magnitude
u_np = u_full[:, 0].detach().cpu().numpy()
v_np = u_full[:, 1].detach().cpu().numpy()
disp_mag = np.sqrt(u_np**2 + v_np**2)

# Get the coordinates for plotting
xx = x_full[:, 0].detach().cpu().numpy()
yy = x_full[:, 1].detach().cpu().numpy()
connect = (t_connect - 1).detach().cpu().numpy()  # Adjust connectivity indices if needed
triang = mtri.Triangulation(xx, yy, connect)

# Create a plot for σ11
fig, ax = plt.subplots(figsize=(6, 5))
cmap = 'jet'
im = ax.tripcolor(triang, sigma11, cmap=cmap, shading='flat', vmin=0, vmax=0.7)
ax.set_title("σ₁₁")
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig.colorbar(im, ax=ax)
plt.tight_layout()

plot_path = os.path.join(results_folder, 'predicted_plot.png')
plt.savefig(plot_path)
plt.close()


# Assume that the following variables are already defined:
# xx, yy : coordinates of collocation points (numpy arrays)
# u_np, v_np : displacement components at each point (numpy arrays)
# results_folder : the folder path to save plots

# Create a high-resolution figure (DPI=300) and larger figure size if desired
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Plot displacement arrows using quiver with larger arrows
# Adjust the scale parameter to 0.5 to make the arrows larger relative to the data units,
# and tweak width and head parameters as needed.
ax.quiver(xx, yy, u_np, v_np, angles='xy', scale_units='xy', scale=0.2,
          color='blue', width=0.005, headwidth=4, headlength=6, headaxislength=5)
# ax.quiver(xx, yy, u_np, v_np, angles='xy', scale_units='xy', scale=1, color='blue', width=0.002)

ax.set_title("Displacement Field")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()

# Save the displacement plot
displacement_plot_path = os.path.join(results_folder, 'displacement_plot.png')
plt.savefig(displacement_plot_path)
plt.close()

# Convert the true displacement tensor to numpy arrays
u_truth = disp_truth[:, 0].detach().cpu().numpy()
v_truth = disp_truth[:, 1].detach().cpu().numpy()

# Extract the x and y coordinates from x_full
xx = x_full[:, 0].detach().cpu().numpy()
yy = x_full[:, 1].detach().cpu().numpy()

# Create a high-resolution figure for the true displacement field
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Plot the true displacement arrows using quiver
# Adjust scale parameters to control arrow size as needed
ax.quiver(xx, yy, u_truth, v_truth, angles='xy', scale_units='xy', scale=0.2,
          color='red', width=0.005, headwidth=4, headlength=6, headaxislength=5)

ax.set_title("True Displacement Field")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()

# Save the plot to a file
os.makedirs(results_folder, exist_ok=True)
true_disp_plot_path = os.path.join(results_folder, 'true_displacement_plot.png')
plt.savefig(true_disp_plot_path)
plt.close()

# Log the contour plot image to wandb
wandb.log({"predicted_stress_plot": wandb.Image(plot_path)})