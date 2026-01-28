import torch
from torchdyn.models import NeuralODE
from torchdyn.core import ODEProblem
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import torch.optim as optim
import pdb
import os
import shutil

from sdf import load_png_to_binary_image, fast_marching_method, apply_sharp_sigmoid, SDF_Loss_Interpolated

# Function to get the latest experiment number
def get_latest_experiment_number(base_dir='experiments'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 0
    existing_experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not existing_experiments:
        return 0
    latest_exp_num = max(int(d.split('_')[1]) for d in existing_experiments if d.startswith('exp_'))
    return latest_exp_num

# Create a new experiment folder
exp_base_dir = 'experiments'
latest_exp_num = get_latest_experiment_number(exp_base_dir)
new_exp_num = latest_exp_num + 1
new_exp_dir = os.path.join(exp_base_dir, f'exp_{new_exp_num:03d}')
os.makedirs(new_exp_dir)

# Save the current script to the experiment folder
script_name = 'main.py'
shutil.copy(__file__, os.path.join(new_exp_dir, script_name))

# Load the PNG image and convert to binary
png_path = "/home/progyan.das/flow/maze.jpg"
binary_image = load_png_to_binary_image(png_path)

# Calculate the 2D SDF using Fast Marching Method
sdf_2d = fast_marching_method(binary_image)

# sdf_2d_sigmoid = -1 * apply_sharp_sigmoid(sdf_2d, k = 2)
sdf_2d_sigmoid = -1 * sdf_2d
print(sdf_2d.max(), sdf_2d.min())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

sdf_loss_function_interpolated = SDF_Loss_Interpolated(sdf_2d_sigmoid, device=device)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 24),
            nn.Tanh(),
            nn.Linear(24, 3)
        )

    def forward(self, x):
        L, sdf_values = sdf_loss_function_interpolated(x[:, 0], x[:, 1])
        neural_net_output = self.net(x)
        modified_output_ = neural_net_output * sdf_values
        modified_output = modified_output_.clone()
        modified_output[:, -1] = modified_output_[:, -1]**2
        return modified_output

f = CustomModel()

print('Using {} device'.format(device))
model = NeuralODE(f, sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4).to(device)

t_span = torch.linspace(0, 1, 100)

# Define points A, B, C, and D
radius = 3.5  # Radius of the smaller circle
A = torch.tensor([140, 140, 0.], dtype=torch.float32).to(device)  # Point on one side of the circle
D = torch.tensor([200, 120], dtype=torch.float32).to(device)  # Diametrically opposite point
C = torch.tensor([140, 180, 0.], dtype=torch.float32).to(device)  # Point on top of the circle
B = torch.tensor([200, 200], dtype=torch.float32).to(device)  # Diametrically opposite point

A_tuple = (A[0].item(), A[1].item())
B_tuple = (B[0].item(), B[1].item())
C_tuple = (C[0].item(), C[1].item())
D_tuple = (D[0].item(), D[1].item())

points = [A_tuple, B_tuple, C_tuple, D_tuple]

plt.imshow(sdf_2d_sigmoid, cmap='RdBu')
plt.title('Signed Distance Function (Fast Marching Method)')
plt.colorbar(label='Distance')

point_labels = ['A', 'B', 'C', 'D']

for label, point in zip(point_labels, points):
    plt.scatter(point[1], point[0], color='yellow', s=100, edgecolor='black', marker='o')  # Note: (y, x) because imshow expects (row, col)
    plt.annotate(label, (point[1], point[0]), textcoords="offset points", xytext=(5,-5), ha='center')

plt.axis('off')
plt.savefig(os.path.join(new_exp_dir, 'sdf_with_points_annotated.png'))

# Optimizers
optimizer_A_B = optim.Adam(model.parameters(), lr=0.01)
optimizer_C_D = optim.Adam(model.parameters(), lr=0.01)

scheduler_A_B = optim.lr_scheduler.StepLR(optimizer_A_B, step_size=100, gamma=0.1)
scheduler_C_D = optim.lr_scheduler.StepLR(optimizer_C_D, step_size=100, gamma=0.1)

trajectories_A = []
trajectories_C = []

# Training loop
n_epochs = 1500  # Number of epochs
trajectories_A = []
trajectories_C = []

def save_plot(epoch, trajectories_A, trajectories_C):
    fig, ax = plt.subplots()
    sdf_plot = ax.imshow(sdf_2d_sigmoid, cmap='RdBu', extent=[0, sdf_2d_sigmoid.shape[1], 0, sdf_2d_sigmoid.shape[0]])
    ax.set_title(f'Signed Distance Function (Epoch {epoch})')
    plt.colorbar(sdf_plot, ax=ax, label='Distance')

    points = {'A': A_tuple, 'D': D_tuple, 'C': C_tuple, 'B': B_tuple}

    for label, (y, x) in points.items():
        ax.scatter(x, y, color='yellow', s=100, edgecolor='black', marker='o')
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, -5), ha='center')

    colors = ['r', 'g', 'b', 'm', 'c']
    for i, traj in enumerate(trajectories_A):
        ax.plot(traj[:, 0, 1], traj[:, 0, 0], colors[i % len(colors)], label=f'Traj A (epoch {i * 20})')
    
    for i, traj in enumerate(trajectories_C):
        ax.plot(traj[:, 0, 1], traj[:, 0, 0], colors[i % len(colors)], label=f'Traj C (epoch {i * 20})')

    plt.legend()
    plt.axis('off')
    plt.savefig(os.path.join(new_exp_dir, f'sdf_particles_traj_epoch_{epoch}.png'))
    plt.close()

try:
    for epoch in range(n_epochs):
        # Training for A to B
        optimizer_A_B.zero_grad()
        traj_A = model.trajectory(A.unsqueeze(0), t_span)
        traj_C = model.trajectory(C.unsqueeze(0), t_span)
        
        loss_A_B = torch.nn.functional.mse_loss(traj_A[-1][:, :-1], B)
        loss_C_D = torch.nn.functional.mse_loss(traj_C[-1][:, :-1], D)

        if loss_A_B > 2 * loss_C_D:
            loss = loss_A_B + 5 * loss_C_D
        elif loss_C_D > 2 * loss_A_B:
            loss = 5 * loss_A_B + loss_C_D
        else:
            loss = loss_A_B + loss_C_D

        loss.backward()
        optimizer_A_B.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss A to B: {loss_A_B.item()}, Loss C to D: {loss_C_D.item()}")
            
        if epoch % 20 == 0:  # Save the trajectories for every 20th epoch
            trajectories_A.append(traj_A.cpu().detach().numpy())
            trajectories_C.append(traj_C.cpu().detach().numpy())
            save_plot(epoch, trajectories_A, trajectories_C)

        scheduler_A_B.step()
        scheduler_C_D.step()

except KeyboardInterrupt:
    pdb.set_trace()
    pass
