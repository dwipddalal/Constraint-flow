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

# Define points A, B, C, and D in 3D
radius = 3.5  # Radius of the smaller circle
A = torch.tensor([2.5, 0, 2], dtype=torch.float32).to(device)  # Point on one side of the circle with height 5
B = torch.tensor([-3.5, 0, -2], dtype=torch.float32).to(device)  # Diametrically opposite point with height 5
C = torch.tensor([0, -2.5, 3], dtype=torch.float32).to(device)  # Point on top of the circle with height 7
D = torch.tensor([0, 3.5, -3], dtype=torch.float32).to(device)  # Diametrically opposite point with height 7

# Optimizers
optimizer_A_B = optim.Adam(model.parameters(), lr=0.01)
optimizer_C_D = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 700  # Number of epochs
for epoch in range(n_epochs):
    # Training for A to B
    optimizer_A_B.zero_grad()
    traj_A = model.trajectory(A.unsqueeze(0), t_span)
    loss_A_B = torch.nn.functional.mse_loss(traj_A[-1], B)
    loss_A_B.backward()
    optimizer_A_B.step()
    
    # Training for C to D
    optimizer_C_D.zero_grad()
    traj_C = model.trajectory(C.unsqueeze(0), t_span)
    loss_C_D = torch.nn.functional.mse_loss(traj_C[-1], D)
    loss_C_D.backward()
    optimizer_C_D.step()

    # Print loss every few epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss A to B: {loss_A_B.item()}, Loss C to D: {loss_C_D.item()}")

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

# Plot trajectories
traj_A_np = traj_A.detach().cpu().numpy().squeeze(1)
traj_C_np = traj_C.detach().cpu().numpy().squeeze(1)

ax.plot(traj_A_np[:, 0], traj_A_np[:, 1], traj_A_np[:, 2], linewidth=4, color="black", label='A to B')
ax.plot(traj_C_np[:, 0], traj_C_np[:, 1], traj_C_np[:, 2], linewidth=4, color="yellow", label='C to D')

# Plot inner and outer shells
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_inner = 2 * np.outer(np.cos(u), np.sin(v))
y_inner = 2 * np.outer(np.sin(u), np.sin(v))
z_inner = 2 * np.outer(np.ones(np.size(u)), np.cos(v))

x_outer = 5 * np.outer(np.cos(u), np.sin(v))
y_outer = 5 * np.outer(np.sin(u), np.sin(v))
z_outer = 5 * np.outer(np.ones(np.size(u)), np.cos(v))

# Inner shell (solid)
ax.plot_surface(x_inner, y_inner, z_inner, color='red', alpha=1.0)

# Outer shell (transparent)
ax.plot_surface(x_outer, y_outer, z_outer, color='red', alpha=0.3)

# Annotate points
ax.text(A[0].item(), A[1].item(), A[2].item(), 'A', color='black')
ax.text(B[0].item(), B[1].item(), B[2].item(), 'B', color='black')
ax.text(C[0].item(), C[1].item(), C[2].item(), 'C', color='yellow')
ax.text(D[0].item(), D[1].item(), D[2].item(), 'D', color='yellow')

ax.legend()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 10])

# Save snapshots from multiple angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]
output_dir = '3d_plots'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for angle in angles:
    ax.view_init(30, angle)
    plt.savefig(f'{output_dir}/plot_angle_{angle}.png')
