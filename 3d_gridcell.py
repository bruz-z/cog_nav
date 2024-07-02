import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class GridCell3D:
    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma
        self.activity = np.zeros((size, size, size))

    def local_excitation(self):
        # Create a 3D Gaussian distribution for local excitation
        center = self.size // 2
        x, y, z = np.indices((self.size, self.size, self.size))
        gaussian = np.exp(-((x-center)**2 + (y-center)**2 + (z-center)**2) / (2*self.sigma**2))
        gaussian /= gaussian.sum()  # Normalize the Gaussian
        self.activity = gaussian

    def global_inhibition(self):
        # Apply global inhibition uniformly
        inhibition = self.activity.sum() / self.activity.size
        self.activity -= inhibition
        self.activity[self.activity < 0] = 0  # Ensure non-negative values

    def normalize_activity(self):
        # Normalize the activity to keep the total activity to 1
        total_activity = self.activity.sum()
        if total_activity > 0:
            self.activity /= total_activity

    def update_activity(self, velocity):
        # Shift activity based on velocity (simple example)
        shift_x, shift_y, shift_z = velocity
        self.activity = np.roll(self.activity, shift_x, axis=0)
        self.activity = np.roll(self.activity, shift_y, axis=1)
        self.activity = np.roll(self.activity, shift_z, axis=2)

    def visualize_activity(self):
        # Visualize the 3D activity (sum over one axis for simplicity)
        plt.imshow(self.activity.sum(axis=2), cmap='viridis')
        plt.colorbar(label='Activity')
        plt.title('3D Grid Cell Activity')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

# 创建并初始化 3D 网格细胞模型
grid_size = 50
sigma = 5.0
grid_cell = GridCell3D(grid_size, sigma)

# 局部兴奋
grid_cell.local_excitation()
grid_cell.visualize_activity()

# 全局抑制
grid_cell.global_inhibition()
grid_cell.normalize_activity()
grid_cell.visualize_activity()

# 更新活动（路径积分）
velocity = (1, 1, 1)  # 示例速度
grid_cell.update_activity(velocity)
grid_cell.visualize_activity()
