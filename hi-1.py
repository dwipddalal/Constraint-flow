import torch
from torchdyn.models import NeuralODE
from torchdyn.core import ODEProblem
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.colors import Normalize

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def sdf_unit_circle(x, y, z):
    return torch.sqrt(x**2 + y**2) - 4

def sdf_disc(x, y, z):
    return (torch.sqrt(x**2 + y**2) - 1) * (torch.sqrt(x**2 + y**2) - 0.25)

def sdf_disc_no_z(x, y):
    return (np.sqrt(x**2 + y**2) - 1) * (np.sqrt(x**2 + y**2) - 0.25)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 24),
            nn.Tanh(),
            nn.Linear(24, 3)
        )

    def forward(self, x):
        sdf_values = sdf_disc(x[:, 0], x[:, 1], x[:, 2]).unsqueeze(1)
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
A = torch.tensor([50, 200, 0.], dtype=torch.float32).to(device)  # Point on one side of the circle
B = torch.tensor([100, 100], dtype=torch.float32).to(device)  # Diametrically opposite point
C = torch.tensor([150, 150, 0.], dtype=torch.float32).to(device)  # Point on top of the circle
D = torch.tensor([200, 200], dtype=torch.float32).to(device)  # Diametrically opposite point


# Optimizers
optimizer_A_B = optim.Adam(model.parameters(), lr=0.01)
optimizer_C_D = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 1000  # Number of epochs
try:
    for epoch in range(n_epochs):
        # Training for A to B
        optimizer_A_B.zero_grad()
        traj_A = model.trajectory(A.unsqueeze(0), t_span)
        traj_C = model.trajectory(C.unsqueeze(0), t_span)
        import pdb
        loss_A_B = torch.nn.functional.mse_loss(traj_A[-1][:, :-1], B)
        loss_C_D = torch.nn.functional.mse_loss(traj_C[-1][:, :-1], D)

        if loss_A_B > 2*loss_C_D:
            loss = loss_A_B + 5*loss_C_D
        elif loss_C_D > 2*loss_A_B:
            loss = 5*loss_A_B + loss_C_D
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
except:
    plt.plot(traj_A.squeeze(1).detach()[:, 0],traj_A.squeeze(1).detach()[:, 1])
    plt.plot(traj_C.squeeze(1).detach()[:, 0],traj_C.squeeze(1).detach()[:, 1])

    trajectory = np.array(traj_A.detach()).squeeze(1)
    trajectory2 = np.array(traj_C.detach()).squeeze(1)
    plt.show()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set up the plot limits
    x_min, x_max = -1.5, 1.5  # Adjust based on radius 1 cylinder
    y_min, y_max = -1.5, 1.5  # Adjust based on radius 1 cylinder
    z_min = 0
    z_max = max(trajectory[:, 2].max(), trajectory2[:, 2].max())  # Use the longest time as height

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Time)')
    ax.set_title('3D Trajectories in Cylinder')

    # Create a cylinder of radius 1
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(z_min, z_max, 100)
    Theta, Z = np.meshgrid(theta, z)
    X = np.cos(Theta)
    Y = np.sin(Theta)

    # Plot the translucent cylinder
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    # Plot the top and bottom circles
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, z_max, color='blue', alpha=0.5)
    ax.plot(x, y, z_min, color='red', alpha=0.5)

    # Prepare color maps for trajectories
    norm1 = Normalize(z_min, z_max)
    norm2 = Normalize(z_min, z_max)
    cmap1 = plt.get_cmap('viridis')
    cmap2 = plt.get_cmap('plasma')

    # Plot trajectories
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            color='blue', linewidth=2, label='Trajectory 1')
    ax.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], 
            color='red', linewidth=2, label='Trajectory 2')

    # Add colorbar for trajectories
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    cbar1 = fig.colorbar(sm1, ax=ax, location='left', label='Time (Trajectory 1)')
    cbar2 = fig.colorbar(sm2, ax=ax, location='right', label='Time (Trajectory 2)')

    # Add legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig('3d_cylinder_trajectories_simple.png', dpi=300, bbox_inches='tight')

# plt.plot(traj_A.squeeze(1).detach()[:, 0],traj_A.squeeze(1).detach()[:, 1])
# plt.plot(traj_C.squeeze(1).detach()[:, 0],traj_C.squeeze(1).detach()[:, 1])

trajectory = np.array(traj_A).squeeze(1)


fig, ax = plt.subplots(figsize=(10, 8))

# Set the limits of the plot
ax.set_xlim(trajectory[:, 0].min(), trajectory[:, 0].max())
ax.set_ylim(trajectory[:, 1].min(), trajectory[:, 1].max())

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Trajectory with Time Color Gradient')

# Prepare the color map
norm = Normalize(trajectory[:, 2].min(), trajectory[:, 2].max())
cmap = plt.get_cmap('viridis')

# Create a line collection object
points = trajectory[:, :2].reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(trajectory[:, 2])
lc.set_linewidth(2)

# Add the line collection to the plot
line = ax.add_collection(lc)

# Add a colorbar
cbar = fig.colorbar(line, ax=ax)
cbar.set_label('Time')

# Initialize the plot
def init():
    line.set_segments([])
    return line,

# Update function for animation
def update(frame):
    line.set_segments(segments[:frame])
    return line,

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=len(trajectory)-1,
                            init_func=init, blit=True, interval=50)

# Show the plot
plt.show()

pdb.set_trace()
# Evaluate vector field
n_pts = 100
x = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
y = torch.linspace(-5, 5, n_pts)  # Set range from -5 to 5
X, Y = torch.meshgrid(x, y)
z = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], 1)
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
plt.savefig('multi_agent_path_planning.png')
plt.show()