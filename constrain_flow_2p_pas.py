import torch
from torchdyn.models import NeuralODE
from torchdyn.core import ODEProblem
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def sdf_unit_circle(x, y, z=0):  # Added default value for z
    return torch.sqrt(x**2 + y**2) - 4

def sdf_disc(x, y, z=0):  # Added default value for z
    return (torch.sqrt(x**2 + y**2) - 5) * (torch.sqrt(x**2 + y**2) - 2)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        if x.shape[1] < 3:
            x = torch.cat((x, torch.zeros(x.shape[0], 3 - x.shape[1], device=x.device)), dim=1)
        sdf_values = sdf_unit_circle(x[:, 0], x[:, 1], x[:, 2]).unsqueeze(1)
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
A = torch.tensor([0., 0., 0.], dtype=torch.float32).to(device)  # Point on one side of the circle
B = torch.tensor([1., 1., 0.], dtype=torch.float32).to(device)  # Added third dimension
C = torch.tensor([0., 1., 0.], dtype=torch.float32).to(device)  # Point on top of the circle
D = torch.tensor([1., 0., 0.], dtype=torch.float32).to(device)  # Added third dimension

# Optimizers
optimizer_A_B = optim.Adam(model.parameters(), lr=0.01)
optimizer_C_D = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 1  # Number of epochs
# try:
for epoch in range(n_epochs):   
    # Training for A to B
    optimizer_A_B.zero_grad()
    traj_A = model.trajectory(A.unsqueeze(0), t_span)
    traj_C = model.trajectory(C.unsqueeze(0), t_span)
    import pdb
    loss_A_B = torch.nn.functional.mse_loss(traj_A[-1][:, :-1], B)
    loss_C_D = torch.nn.functional.mse_loss(traj_C[-1][:, :-1], D)
    if loss_A_B > 2*loss_C_D:
        loss = loss_A_B + 2*loss_C_D
    elif loss_C_D > 2*loss_A_B:
        loss = 2*loss_A_B + loss_C_D
    else:
        loss = loss_A_B + loss_C_D

    loss.backward()
    optimizer_A_B.step()
    
    # Training for C to D
    # optimizer_C_D.zero_grad()

    
    # loss_C_D.backward()
    # optimizer_C_D.step()

    # Print loss every few epochs
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss A to B: {loss_A_B.item()}, Loss C to D: {loss_C_D.item()}")
# except:
#     plt.plot(traj_A.squeeze(1).detach()[:, 0],traj_A.squeeze(1).detach()[:, 1])
#     plt.plot(traj_C.squeeze(1).detach()[:, 0],traj_C.squeeze(1).detach()[:, 1])
#     plt.show()

# pdb.set_trace()
# Evaluate vector field

print('it came here')
n_pts = 100
x = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
y = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
X, Y = torch.meshgrid(x, y)
z = torch.cat([X.reshape(-1,1), Y.reshape(-1,1), torch.zeros(X.numel(), 1)], 1)  # Added third dimension
f = model.vf(0, z.to(device)).cpu().detach()
fx, fy = f[:,0], f[:,1]
fx, fy = fx.reshape(n_pts, n_pts), fy.reshape(n_pts, n_pts)

# Plot vector field and its intensity
fig = plt.figure(figsize=(10, 5))  # Set figure size to 10x5 inches
ax = fig.add_subplot(111)

# Vector field
ax.streamplot(X.numpy().T, Y.numpy().T, fx.numpy().T, fy.numpy().T, color='black', density=0.5)

# Contour plot with a continuous color map (e.g., 'viridis')
contour = ax.contourf(X.T, Y.T, torch.sqrt(fx.T**2 + fy.T**2), cmap='jet')
plt.plot(traj_A.detach().cpu().squeeze(1)[:, 0], traj_A.detach().cpu().squeeze(1)[:, 1], linewidth=4, color="white", label='A to B')
plt.plot(traj_C.detach().cpu().squeeze(1)[:, 0], traj_C.detach().cpu().squeeze(1)[:, 1], linewidth=4, color="yellow", label='C to D')

# Set axis limits to [-5, 5] in both directions
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Increase the width of the axis lines
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

# Add a circle of radius 2
circle = Circle((0, 0), 2, color='red', fill=False, linewidth=10)
ax.add_patch(circle)

# Add a circle of radius 5
circle2 = Circle((0, 0), 5, color='red', fill=False, linewidth=10)
ax.add_patch(circle2)

# Plot start and end points and trajectories
plt.plot([A[0].cpu(), B[0].cpu()], [A[1].cpu(), B[1].cpu()], linestyle="--", linewidth=4, color="grey")
plt.plot([C[0].cpu(), D[0].cpu()], [C[1].cpu(), D[1].cpu()], linestyle="--", linewidth=4, color="grey")
plt.scatter(traj_A.detach().cpu().squeeze(1)[0, 0], traj_A.detach().cpu().squeeze(1)[0, 1], color="white", s=40)
plt.scatter(traj_A.detach().cpu().squeeze(1)[-1, 0], traj_A.detach().cpu().squeeze(1)[-1, 1], color="white", s=40)
plt.scatter(traj_C.detach().cpu().squeeze(1)[0, 0], traj_C.detach().cpu().squeeze(1)[0, 1], color="yellow", s=40)
plt.scatter(traj_C.detach().cpu().squeeze(1)[-1, 0], traj_C.detach().cpu().squeeze(1)[-1, 1], color="yellow", s=40)

# Annotate points
plt.annotate(f'A ({A[0].item()}, {A[1].item()})', (A[0].item(), A[1].item()), textcoords="offset points", xytext=(10,-10), ha='center', color="white")
plt.annotate(f'B ({B[0].item()}, {B[1].item()})', (B[0].item(), B[1].item()), textcoords="offset points", xytext=(10,-10), ha='center', color="white")
plt.annotate(f'C ({C[0].item()}, {C[1].item()})', (C[0].item(), C[1].item()), textcoords="offset points", xytext=(10,-10), ha='center', color="yellow")
plt.annotate(f'D ({D[0].item()}, {D[1].item()})', (D[0].item(), D[1].item()), textcoords="offset points", xytext=(10,-10), ha='center', color="yellow")

plt.legend()
plt.savefig('multi_agent_path_planning_2d.png')
# plt.show()
