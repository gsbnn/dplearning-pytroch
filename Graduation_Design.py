import torch
from torch import nn
import pandas as pd

def get_dataset(data_file_path, select_channels, label):
    """获取数据集"""
    data = pd.read_excel(data_file_path, header=0, index_col=None, usecols=select_channels) # 忽略采样序号和热流速度
    target_data = torch.tensor(data[label].values).reshape(-1, 1) # torch.Size([20000, 1])
    input_data = torch.tensor(data.drop(label, axis=1).values) # torch.Size([20000, 15])
    dataset = torch.cat([target_data, input_data], 1) # torch.Size([20000, 16])
    return dataset

def batch_iter(dataset, batch_size):
    """产生批数据"""
    for i in range((len(dataset) + batch_size - 1) // batch_size):
        index = i * batch_size
        if index + batch_size > len(dataset)-1:
            yield dataset[index:, 1:], dataset[index:, 0] # torch.Size([batch_size, 15]) torch.Size([batch_size])
        else:
            yield dataset[index: index+batch_size, 1:], dataset[index: index+batch_size, 0]

def create_data_win(data, win_size):
    """产生窗口数据"""
    for i in range(len(data) - win_size + 1):
        yield data[i: i+win_size]

def get_distance(data, curret_sample):
    """计算窗口样本的距离之和"""
    close_fn = torch.abs(curret_sample - data)
    close_fn = torch.exp(-1* close_fn)
    close_fn = close_fn.sum(dim=1) / data.shape[1]
    distance = torch.sum(1 / (close_fn + 0.01) -1, dim=0, keepdim=True)
    return distance

def get_similar_data(dataset, batch_size, win_size, curret_sample):
    """获取历史相似样本"""
    similar_data = []
    index_list = []
    for data, labels in batch_iter(dataset, batch_size):
        distance_list = []
        # 数据标准化
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data_norm = (data - data_mean)/data_std
        curret_sample_norm = (curret_sample - data_mean)/data_std
        # 获取相似样本
        for win_data in create_data_win(data_norm, win_size):
            distance = get_distance(win_data, curret_sample_norm)
            distance_list.append(distance)
        distances = torch.cat(distance_list)
        index = distances.argmin(dim=0) # 每个批次的最近似窗口数据
        similar_data.append((data[index: index+win_size].t(), 
                             labels[index+win_size-1]))
        index_list.append(index)
    return index_list, similar_data # similar_data: list[(data[nodes, in_feature], labels[in_features])]

if __name__ == '__main__':

    data_file_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
    select_channels = "B:N,P:Q"
    label = "产物浓度"
    dataset = get_dataset(data_file_path, select_channels, label)
    curret_sample = dataset[22*400+1, 1:]
    print("当前样本：", curret_sample)

    batch_size = 400
    win_size = 5
        
    index_list, similra_data = get_similar_data(dataset, batch_size, win_size, curret_sample)
    print(index_list, len(index_list), similra_data[1][1])

#    data_list = []
#    for data, _ in batch_iter(dataset, batch_size):
#        for data_win in create_data_win(data, 4):
#            data_list.append(data_win)
#        break
#    print(data_list[-1])

#    print(len(similra_data))
#    print(index_con[0:5])
#    print(similra_data[0:5])
#    a = 1/torch.arange(1, 10)
#    b = a.sum(dim=0, keepdim=True)
#    print(a,b)
#    b = torch.arange(0, 15).sum(dim=0, keepdim=True)
#    c = [a, b]
#    print(torch.cat(c))
#    print(a)



#    for data, label in batch_iter(dataset, 400):
#        data_mean = data.mean(dim=0)
#        data_std = data.std(dim=0)
#        print("样本均值：\r", data_mean)
#        print(data_std)
#        print(data[1])
#        data_norm = (data - data_mean)/data_std
#        curret_sample_norm = (curret_sample - data_mean)/data_std
#        print("标准化当前样本：", curret_sample_norm)
#        print(data_norm[1])
#        cov = data_norm[77: 82].t().cov()
#        cov_inv = cov.inverse()
#        MahaMatrix = torch.matmul(curret_sample_norm-data_norm[77: 82], cov_inv)
#        Mahadistance = torch.diag(torch.matmul(MahaMatrix, (curret_sample_norm-data_norm[77: 82]).t()))
#        Mahadistance = torch.sqrt(Mahadistance)
#        print("协方差矩阵：\n", cov)
#        print("马氏矩阵：\n",Mahadistance)
#        print(cov_inv)
#        break