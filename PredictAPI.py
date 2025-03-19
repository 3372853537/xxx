import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import re

#————————————————————————————————————————————————————————————————————————————————————————
#数据处理部分

# 定义文件路径
file_path = r'MoviesData\dailyPredictData\BoxingData.txt'
#利用gpu进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 用于存储所有电影数据的列表
all_movies_data = []

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 提取电影名
        movie_name = line.split(' ', 1)[0]
        # 提取每日数据
        daily_data_strs = re.findall(r'\[.*?\]', line)
        movie_data = []
        for daily_data_str in daily_data_strs:
            # 去除方括号并分割数据
            daily_data = [float(x) for x in daily_data_str.strip('[]').split(',')]
            movie_data.append(daily_data)
        all_movies_data.append(movie_data)

# 将列表转换为三维 NumPy 数组
data_array = np.array(all_movies_data)

# 初始化 MinMaxScaler
scaler = MinMaxScaler()

# 将数据调整为二维数组进行归一化
data_2d = data_array.reshape(-1, data_array.shape[-1])
scaled_data_2d = scaler.fit_transform(data_2d)

scaled_data_array = scaled_data_2d.reshape(data_array.shape)

# 将归一化后的 NumPy 数组转换为 PyTorch 张量
tensor_data = torch.from_numpy(scaled_data_array).float()


# 将 NumPy 数组转换为 PyTorch 张量  （批量大小，时间步，特征数） torch.Size([344, 30, 4])
#tensor_data = torch.from_numpy(data_array).float()
#测试集是最后20个
test_data_size = 20

#批量大小
batch_size = tensor_data.shape[1]
#根据test_data_size，划分测试集和训练集
#torch.Size([324, 30, 4])
#torch.Size([20, 30, 4])
train_data = tensor_data[:-test_data_size]
test_data = tensor_data[-test_data_size:]

#转置成（时间步，批量大小，特征数）
#torch.Size([30, 324, 4])
#torch.Size([30, 20, 4])
transposed_train_tensor = torch.transpose(train_data, 0, 1).to(device)
transposed_test_tensor = torch.transpose(test_data, 0, 1).to(device)
#预测窗口，从前多少天预测
train_window = 15

#创建训练的input （train_seq）和 labels（train_label）
#（train_seq）  大小（train_window，batch_size,features） 
#（train_label）大小（1，batch_size,features）
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(transposed_train_tensor, train_window)
train_inout_seq = [(seq.to(device), labels.to(device)) for seq, labels in train_inout_seq]
#——————————————————————————————————————————————————————————————————————————————————————————
#训练部分

#搭建模型
class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=1500, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        #定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        #定义全连接层
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # 获取批量大小
        batch_size = input_seq.size(1)
        # 初始化隐藏状态和细胞状态，形状为 (num_layers, batch_size, hidden_layer_size)
        self.hidden_cell = (
            torch.zeros(1, batch_size, self.hidden_layer_size, device=input_seq.device),
            torch.zeros(1, batch_size, self.hidden_layer_size, device=input_seq.device)
        )
        # 输入序列已经是 (时间步, 批量大小, 输入特征维度) 的形状，无需调整
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        # 调整 LSTM 输出的形状，以便输入到全连接层
        # lstm_out 的形状是 (时间步, 批量大小, 隐藏层维度)
        # 调整为 (时间步 * 批量大小, 隐藏层维度)
        lstm_out_reshaped = lstm_out.view(-1, self.hidden_layer_size)
        # 通过全连接层得到预测结果
        # predictions 的形状是 (时间步 * 批量大小, 输出维度)
        
        predictions = self.linear(lstm_out_reshaped)
        # 重新调整预测结果的形状为 (时间步, 批量大小, 输出维度)
        
        predictions = predictions.view(input_seq.size(0), batch_size, -1)
        # 返回每个批次最后一个时间步的预测结果
        # 形状为 (批量大小, 输出维度)
        return predictions[-1]

#将模型放在gpu上
model = LSTM().to(device)
#均方误差
loss_function = nn.MSELoss()
#Adam优化
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#训练10000轮
epochs = 15000

for i in range(epochs):
    for seq, labels in train_inout_seq:
        #将input和labels放入gpu上
        seq = seq.to(device)  # 直接使用现有张量
        labels = labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(1, seq.size(1), model.hidden_layer_size, device=seq.device),
            torch.zeros(1, seq.size(1), model.hidden_layer_size, device=seq.device)
        )
        #预测值形状为 (批量大小, 1)
        y_pred = model(seq)
        #（labels）大小（1，batch_size,features）  
        #为了和预测值进行比较  调整形状为（batch_size，1）
        labels = labels.squeeze(0)[:, 0].unsqueeze(1)
        single_loss = loss_function(y_pred[:], labels)
        single_loss.backward()
        optimizer.step()
        
        
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
 
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# 保存模型的 state_dict
torch.save(model.state_dict(), 'model.pth')

# 加载模型
loaded_model = LSTM().to(device)
# 显式设置 weights_only=True
loaded_model.load_state_dict(torch.load('model.pth', weights_only=True))
loaded_model.eval()
#——————————————————————————————————————————————————————————————————————————————————
#预测部分

#输入前几天票房数据
#transposed_test_tensor大小为torch.Size([30, 20, 4])
#test_inputs大小为torch.Size([15, 20, 4])
test_inputs = transposed_test_tensor[:train_window,:,:]

#base_inputs是对test_inputs的一个备份
base_inputs= test_inputs

#开启评估模式
loaded_model.eval()
#放到gpu上
transposed_test_tensor = transposed_test_tensor.to(device)
test_inputs = test_inputs.to(device)

#将剩下的几天全部预测出来
for i in range(30-train_window):
    seq = test_inputs[-train_window:].to(device)
    with torch.no_grad():
        loaded_model.hidden_cell = (
            torch.zeros(1, seq.size(1), loaded_model.hidden_layer_size, device=seq.device),
            torch.zeros(1, seq.size(1), loaded_model.hidden_layer_size, device=seq.device)
        )
        #预测值形状为 (批量大小, 1)
        seq_pred=loaded_model(seq)

        #将真实的后面三个特征拼接进去，方便预测
        additional_features = transposed_test_tensor[i+train_window, :, 1:].unsqueeze(0)
        seq_pred = torch.cat((seq_pred, additional_features.squeeze(0)), dim=1)
        #添加到test_inputs后面
        test_inputs = torch.cat((test_inputs, seq_pred.unsqueeze(0)), dim=0)

#test_inputs前几天是真实数据，后是预测数据
prediction_result = test_inputs[:,:,:]
#未进行线性回归的票房结果
prediction_result =  prediction_result[:,:,0]

# 反归一化预测结果
# 提取票房特征在归一化前的最小和最大值
min_val = scaler.data_min_[0]
max_val = scaler.data_max_[0]
denormalized_prediction = prediction_result.cpu().numpy() * (max_val - min_val) + min_val
#——————————————————————————————————————————————————————————————————————————————————————————
#画图部分
# 生成时间步的 x 轴数据
"""
x = torch.arange(denormalized_prediction.shape[0]).cpu().numpy()  # 将 x 移动到 CPU 并转换为 numpy 数组
Test_data = test_data[:, :, 0]
Test_data = torch.transpose(Test_data, 0, 1)
# 遍历每个批量
for i in range(denormalized_prediction.shape[1]):
    # 提取当前批量的数据
    pred_batch = denormalized_prediction[:, i].cpu().numpy()  # 将 pred_batch 移动到 CPU 并转换为 numpy 数组
    test_batch = Test_data[:, i].cpu().numpy()  # 将 test_batch 移动到 CPU 并转换为 numpy 数组
    # 创建一个新的图形
    plt.figure()
    plt.title(f'Comparison of Batch {i}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)

    # 绘制预测结果和测试数据
    plt.plot(x, pred_batch, label='Prediction', color='blue')
    plt.plot(x, test_batch, label='Test Data', color='red')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()
"""


