import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import heapq

def load_png_to_binary_image(png_path, shape=(512, 512)):
    png_image = Image.open(png_path).convert("L")
    png_image = png_image.resize(shape, Image.LANCZOS)
    numpy_image = np.array(png_image)
    _, binary_image = cv2.threshold(numpy_image, 200, 255, cv2.THRESH_BINARY)
    binary_image = binary_image // 255
    return binary_image

def fast_marching_method(binary_image):
    # Initialize SDF with large positive and negative values
    sdf = np.where(binary_image == 1, float('inf'), -float('inf'))

    # Initialize a priority queue and add boundary pixels with zero distance
    heap = []
    for x in range(1, binary_image.shape[0] - 1):
        for y in range(1, binary_image.shape[1] - 1):
            if binary_image[x, y] != binary_image[x + 1, y] or \
               binary_image[x, y] != binary_image[x - 1, y] or \
               binary_image[x, y] != binary_image[x, y + 1] or \
               binary_image[x, y] != binary_image[x, y - 1]:
                sdf[x, y] = 0
                heapq.heappush(heap, (0, (x, y)))

    # Update SDF values using the Fast Marching Method
    while heap:
        distance, (x, y) = heapq.heappop(heap)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                x_neighbor, y_neighbor = x + dx, y + dy
                if 0 <= x_neighbor < binary_image.shape[0] and 0 <= y_neighbor < binary_image.shape[1]:
                    new_distance = distance + np.sqrt(dx ** 2 + dy ** 2)
                    if new_distance < abs(sdf[x_neighbor, y_neighbor]):
                        sdf[x_neighbor, y_neighbor] = new_distance if binary_image[x_neighbor, y_neighbor] == 1 else -new_distance
                        heapq.heappush(heap, (new_distance, (x_neighbor, y_neighbor)))

    return sdf

def apply_sharp_sigmoid(array, k=0.000, x0=0):
    if k == 0:
        return array
    return 1 / (1 + np.exp(-k * (array - x0)))

class SDF_Loss_Interpolated(nn.Module):
    def __init__(self, sdf_array, device='cpu'):
        super(SDF_Loss_Interpolated, self).__init__()
        self.sdf_array = torch.tensor(sdf_array, dtype=torch.float32).to(device)
        self.device = device

    def bilinear_interpolation(self, x, y, report = True):
        x1, y1 = torch.floor(x).long(), torch.floor(y).long()
        x2, y2 = x1 + 1, y1 + 1

        # Bounds checking
        max_x, max_y = self.sdf_array.shape[0] - 1, self.sdf_array.shape[1] - 1
        x1, x2 = torch.clamp(x1, 0, max_x), torch.clamp(x2, 0, max_x)
        y1, y2 = torch.clamp(y1, 0, max_y), torch.clamp(y2, 0, max_y)

        # Interpolation weights
        t = (x - x1.float()).to(self.device)
        u = (y - y1.float()).to(self.device)

        if report:
            # print(f"Interpolation between: {x1.item(), y1.item()} and {x2.item(), y2.item()}")
            # print(f"Interpolation weights: {t.item(), u.item()}")
            # print(f"Interpolation values: {self.sdf_array[x1, y1].item(), self.sdf_array[x2, y1].item(), self.sdf_array[x1, y2].item(), self.sdf_array[x2, y2].item()}")
            pass
        sdf_values = (1 - t) * (1 - u) * self.sdf_array[x1, y1] + \
                     t * (1 - u) * self.sdf_array[x2, y1] + \
                     (1 - t) * u * self.sdf_array[x1, y2] + \
                     t * u * self.sdf_array[x2, y2]

        return sdf_values

    def forward(self, x, y):
        report = True
        if x.shape == torch.Size([]):
            print(x,y)
            report = False
        sdf_values = torch.clamp(self.bilinear_interpolation(x, y, report = report), -1000, 0)

        # Sample loss computation
        loss = torch.sum(sdf_values)

        return loss, sdf_values


# Load the PNG image and convert to binary
png_path = "/home/progyan.das/flow/SDF.png"
binary_image = load_png_to_binary_image(png_path)

# Calculate the SDF using Fast Marching Method
sdf_fmm = fast_marching_method(binary_image)

# Display the SDF
plt.imshow(sdf_fmm, cmap='RdBu')
plt.title('Signed Distance Function (Fast Marching Method)')
plt.axis('off')
plt.colorbar(label='Distance')
# plt.show()
plt.savefig('sdf_check3.png')

# Apply sharp sigmoid to the SDF of the PNG image
sdf_fmm_png_sigmoid = apply_sharp_sigmoid(sdf_fmm, k=0, x0=0)
sdf_fmm_png_sigmoid_sharper = apply_sharp_sigmoid(sdf_fmm, k=1, x0=0)
