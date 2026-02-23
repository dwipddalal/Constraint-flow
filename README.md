# ODESolvers are also Wayfinders: Neural ODEs for Multi-Agent Pathplanning

A framework for multi-agent path planning using Neural ODEs with constraint-based flows. The approach learns smooth, collision-free trajectories by embedding obstacle constraints directly into the neural network dynamics through Signed Distance Functions (SDF).

## Key Idea

Traditional path planning methods often struggle with smooth trajectory generation in complex environments. This work uses **Neural ODEs** to learn continuous-time dynamics that naturally respect obstacle constraints by:

1. Computing a **Signed Distance Function (SDF)** from environment obstacles
2. **Multiplying** neural network outputs by SDF values at each point
3. This ensures the vector field **vanishes at obstacle boundaries**, preventing collisions

```
modified_output = neural_net_output × SDF(x, y)
```

When SDF → 0 (near obstacles), the flow magnitude → 0, creating natural avoidance behavior.

## Results

### 2D Maze Navigation
<img width="510" alt="2D maze navigation with SDF constraints" src="https://github.com/dwipddalal/Constrained-Diffeomorphism-for-Obstacle-Avoidance-and-Multi-Agent-Path-Planning/assets/91228207/db2b4e4c-fef3-41b6-926d-386935a7234b">

### Multi-Agent Coordination
<img width="523" alt="Multi-agent path planning" src="https://github.com/dwipddalal/Constrained-Diffeomorphism-for-Obstacle-Avoidance-and-Multi-Agent-Path-Planning/assets/91228207/67674f99-c58f-474b-8ab9-ed7537a35aae">

### 3D Trajectory Planning
<img width="539" alt="3D obstacle avoidance" src="https://github.com/dwipddalal/Constrained-Diffeomorphism-for-Obstacle-Avoidance-and-Multi-Agent-Path-Planning/assets/91228207/701bf95c-61f1-4b7e-85b8-fc027db38f18">

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Neural ODE Model                      │
├─────────────────────────────────────────────────────────┤
│  Input: (x, y, z) or (x, y, t)                          │
│           ↓                                              │
│  ┌─────────────────┐    ┌──────────────────┐            │
│  │  Neural Network │    │  SDF Computation │            │
│  │  Linear → Tanh  │    │  (Fast Marching) │            │
│  │  → Linear       │    │                  │            │
│  └────────┬────────┘    └────────┬─────────┘            │
│           │                      │                       │
│           └──────────┬───────────┘                       │
│                      ↓                                   │
│              output × SDF(x,y)                           │
│                      ↓                                   │
│           Constrained Vector Field                       │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── main.py                      # Main 2D path planning with maze
├── sdf.py                       # SDF computation using Fast Marching Method
├── constrain_flow_2d.py         # 2D constraint flow implementation
├── constrain_flow_3d.py         # 3D with spherical shell obstacles
├── constrain_flow_3d 10 agents.py  # Multi-agent (10) 3D planning
├── constrain_flow_2p_pas.py     # Two-point planning variant
├── NeuralODE_Constrained.ipynb  # Interactive notebook
├── Constrain flow v1/           # Earlier notebook experiments
│   ├── obstacleAvoidance.ipynb
│   ├── codeObstacleAvoidance.ipynb
│   └── NeuralODE_Constrained.ipynb
└── *.png                        # Visualization outputs
```

## How It Works

### 1. SDF Computation
The Signed Distance Function encodes obstacle geometry:
- **Positive values**: Free space (distance to nearest obstacle)
- **Negative values**: Inside obstacles
- **Zero**: Obstacle boundary

```python
from sdf import load_png_to_binary_image, fast_marching_method

binary_image = load_png_to_binary_image("maze.jpg")
sdf_2d = fast_marching_method(binary_image)
```

### 2. Constrained Neural ODE
The neural network output is multiplied by SDF values:

```python
class CustomModel(nn.Module):
    def forward(self, x):
        L, sdf_values = sdf_loss_function(x[:, 0], x[:, 1])
        neural_net_output = self.net(x)
        modified_output = neural_net_output * sdf_values
        return modified_output
```

### 3. Training
Optimize trajectories from start to goal points:

```python
model = NeuralODE(f, sensitivity='adjoint', solver='rk4')

for epoch in range(n_epochs):
    traj_A = model.trajectory(A.unsqueeze(0), t_span)
    loss = F.mse_loss(traj_A[-1], B)  # Target point B
    loss.backward()
    optimizer.step()
```

## Requirements

- Python 3.8+
- PyTorch
- torchdyn
- numpy
- matplotlib
- opencv-python (cv2)
- Pillow

## Usage

### 2D Maze Navigation
```bash
python main.py
```

### 3D Path Planning
```bash
python constrain_flow_3d.py
```

### Multi-Agent Planning
```bash
python "constrain_flow_3d 10 agents.py"
```

## Key Features

- **Guaranteed Collision Avoidance**: SDF multiplication ensures zero velocity at obstacles
- **Smooth Trajectories**: Neural ODEs produce continuous, differentiable paths
- **Multi-Agent Support**: Scales to multiple agents with shared learned dynamics
- **Flexible Obstacles**: Works with arbitrary obstacle shapes via SDF
- **2D and 3D**: Implementations for both planar and spatial planning

## Citation

If you use this code in your research, please cite accordingly.

## License

This project is for research purposes.
