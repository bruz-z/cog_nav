import numpy as np
import matplotlib.pyplot as plt

class HeadDirectionCellNetwork:
    def __init__(self, num_cells, sigma_theta):
        self.num_cells = num_cells
        self.sigma_theta = sigma_theta
        self.activity = np.zeros(num_cells)
        self.preferred_directions = np.linspace(0, 2 * np.pi, num_cells, endpoint=False)
        self.activity[num_cells // 2] = 1  # 初始化中心细胞活动

    def local_excitation(self):
        # Create a Gaussian distribution for local excitation
        u = np.arange(self.num_cells) - self.num_cells // 2
        gaussian = np.exp(-u**2 / (2 * self.sigma_theta**2))
        gaussian /= gaussian.sum()  # Normalize the Gaussian
        return gaussian

    def global_inhibition(self):
        # Apply global inhibition uniformly
        inhibition = self.activity.sum() / self.num_cells
        self.activity -= inhibition
        self.activity[self.activity < 0] = 0  # Ensure non-negative values

    def normalize_activity(self):
        # Normalize the activity to keep the total activity to 1
        total_activity = self.activity.sum()
        if total_activity > 0:
            self.activity /= total_activity

    def update_activity(self, angular_velocity, dt):
        # Update activity based on angular velocity
        shift = int(angular_velocity * dt * self.num_cells / (2 * np.pi))
        self.activity = np.roll(self.activity, shift)
        self.global_inhibition()
        self.normalize_activity()

    def visualize_activity(self, time, angular_velocity):
        # Visualize the activity of the head direction cells
        plt.bar(np.arange(self.num_cells), self.activity, color='r')
        plt.axvline(x=self.num_cells // 2 + int(angular_velocity * time * self.num_cells / (2 * np.pi)), color='b')
        plt.xlabel('Head direction cells')
        plt.ylabel('Activity')
        plt.title(f'Activity at time = {time}s\n$\\theta$ = {angular_velocity * time:.1f} rad')
        plt.show()

# 创建并初始化方向细胞网络
num_cells = 36
sigma_theta = 2.0
hd_network = HeadDirectionCellNetwork(num_cells, sigma_theta)

# 初始局部兴奋
gaussian_excitation = hd_network.local_excitation()
hd_network.activity = np.convolve(hd_network.activity, gaussian_excitation, mode='same')
hd_network.normalize_activity()

# 模拟角速度和不同时间步长下的活动
angular_velocity = 0.2  # 示例角速度，单位：弧度/秒
time_steps = [1, 2, 3, 4]  # 时间步长，单位：秒

for t in time_steps:
    hd_network.update_activity(angular_velocity, t - (time_steps[0] if t != time_steps[0] else 0))
    hd_network.visualize_activity(t, angular_velocity)
