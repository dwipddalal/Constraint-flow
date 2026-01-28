import torch
from torchdyn.models import NeuralODE
from torchdyn.core import ODEProblem
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def sdf_elongated_disc(x, y, z):
    return (torch.sqrt(x**2 + y**2 + (z/5)**2) - 5) * (torch.sqrt(x**2 + y**2 + (z/5)**2) - 2)

class CustomModel3D(nn.Module):
    def __init__(self):
        super(CustomModel3D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        sdf_values = sdf_elongated_disc(x[:, 0], x[:, 1], x[:, 2]).unsqueeze(1)
        neural_net_output = self.net(x)
        modified_output = neural_net_output * sdf_values
        return modified_output
f = CustomModel3D()
print('Using {} device'.format(device))
model = NeuralODE(f, sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4).to(device)

t_span = torch.linspace(0, 1, 100)

# Define points for 10 agents (20 points in total)
points = [
    [2.5, 0, 2], [-3.5, 0, -2],
    [0, -2.5, 3], [0, 3.5, -3],
    [1.5, 1.5, 4], [-1.5, -1.5, -4],
    [2, -2, 2.5], [-2, 2, -2.5],
    [3, 1, 1.5], [-3, -1, -1.5],
    [1, 3, 2], [-1, -3, -2],
    [2, 2, 3], [-2, -2, -3],
    [3, 0, 4], [-3, 0, -4],
    [0, 3, 1.5], [0, -3, -1.5],
    [1.5, -1.5, 2], [-1.5, 1.5, -2]
]

# Convert points to tensors
points = [torch.tensor(p, dtype=torch.float32).to(device) for p in points]

# Optimizers
optimizers = [optim.Adam(model.parameters(), lr=0.01) for _ in range(10)]

# Training loop
n_epochs = 700  # Number of epochs
for epoch in range(n_epochs):
    for i in range(10):
        # Training for each pair of points
        A = points[2*i]
        B = points[2*i + 1]
        optimizer = optimizers[i]
        
        optimizer.zero_grad()
        traj = model.trajectory(A.unsqueeze(0), t_span)
        loss = torch.nn.functional.mse_loss(traj[-1], B)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 and i == 0:
            print(f"Epoch {epoch}, Loss for pair {i+1}: {loss.item()}")

# Evaluate vector field
n_pts = 20
x = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
y = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
z = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
X, Y, Z = torch.meshgrid(x, y, z)
grid_points = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], 1)
f = model.vf(0, grid_points.to(device)).cpu().detach()
fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
fx, fy, fz = fx.reshape(n_pts, n_pts, n_pts), fy.reshape(n_pts, n_pts, n_pts), fz.reshape(n_pts, n_pts, n_pts)

# Plot vector field and its intensity
fig = plt.figure(figsize=(15, 10))  # Set figure size to 15x10 inches
ax = fig.add_subplot(111, projection='3d')

# Vector field
ax.quiver(X.numpy(), Y.numpy(), Z.numpy(), fx.numpy(), fy.numpy(), fz.numpy(), length=0.1, normalize=True)

# Plot trajectories for all pairs
for i in range(10):
    A = points[2*i]
    B = points[2*i + 1]
    traj = model.trajectory(A.unsqueeze(0), t_span)
    traj_np = traj.detach().cpu().numpy().squeeze(1)
    
    color = plt.cm.jet(i / 10)  # Different color for each trajectory
    ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], linewidth=4, color=color, label=f'Pair {i+1}')
    ax.text(A[0].item(), A[1].item(), A[2].item(), f'A{i+1}', color=color)
    ax.text(B[0].item(), B[1].item(), B[2].item(), f'B{i+1}', color=color)

# Define u and v for spherical coordinates
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Inner shell
x_inner = 2 * np.outer(np.cos(u), np.sin(v))
y_inner = 2 * np.outer(np.sin(u), np.sin(v))
z_inner = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Outer shell
x_outer = 5 * np.outer(np.cos(u), np.sin(v))
y_outer = 5 * np.outer(np.sin(u), np.sin(v))
z_outer = 25 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot inner shell (solid)
ax.plot_surface(x_inner, y_inner, z_inner, color='red', alpha=1.0)

# Plot outer shell (transparent)
ax.plot_surface(x_outer, y_outer, z_outer, color='red', alpha=0.3)

# Setting axis limits and labels
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-25, 25])
ax.legend(['Inner Shell', 'Outer Shell'])

# Save snapshots from multiple angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]
output_dir = '3d_plots_10'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for angle in angles:
    ax.view_init(30, angle)
    plt.savefig(f'{output_dir}/plot_angle_{angle}.png')

# Generate boundary points for model evaluation
def generate_boundary_points():
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    points = np.array([np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones(np.size(u)), np.cos(v))])
    return points.reshape(3, -1).T  # reshaping for easier handling

inner_points = torch.tensor(generate_boundary_points() * 2, dtype=torch.float32)  # adjusting scale for inner
outer_points = torch.tensor(generate_boundary_points() * 5, dtype=torch.float32)  # adjusting scale for outer

# Assume 'model' and 'device' are defined elsewhere (model must be loaded and device set)
inner_points, outer_points = inner_points.to(device), outer_points.to(device)

# Evaluate model on boundary points
inner_output = model.vf(0, inner_points)
outer_output = model.vf(0, outer_points)

# Print mean and std of the outputs to check if they are close to zero
print("Inner Ellipsoid Boundary Output - Mean:", inner_output.mean().item(), "Std:", inner_output.std().item())
print("Outer Ellipsoid Boundary Output - Mean:", outer_output.mean().item(), "Std:", outer_output.std().item())
