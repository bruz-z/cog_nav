import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    # 读取数据
    file_path = 'path_to_your_csv_file/airsim_rec.txt'  # 修改为实际文件路径
    data = pd.read_csv(file_path, delimiter='\t')

    # 提取相关列
    data_extracted = data[['POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z']]

    # 过滤掉POS_X为0或43.8475的行
    filtered_data = data_extracted[(data_extracted['POS_X'] != 0) & (data_extracted['POS_X'] != 43.8475)]

    # 准备数据
    X = filtered_data[['Q_W', 'Q_X', 'Q_Y', 'Q_Z']].values
    Y = filtered_data[['POS_X', 'POS_Y', 'POS_Z']].values

    # 数据归一化
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # 拆分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # 转换为PyTorch的Tensor并调整形状
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [samples, timesteps, features]
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型参数
    input_size = 4
    hidden_size = 50
    output_size = 3
    model = LSTMModel(input_size, hidden_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并记录训练损失
    train_losses = []

    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 可视化训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
