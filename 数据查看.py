import os
import pandas as pd
import random

# 设定txt文件所在的文件夹路径
txt_folder_path = 'data/pensim1000批次'  # 修改为你的txt文件夹路径
output_excel_path = 'data/pensimdata_full_variable_100_batch_本科.xlsx'  # 输出的Excel文件路径

# 预定义列名
columns =["采样时刻","曝气率","搅拌速率","底物流加速度","底物流加温度","底物浓度","溶解氧浓度","菌体浓度","产物浓度","培养液体积","CO2浓度","PH值","反应罐温度","反应热","酸的流速","碱的浓度","冷水流速","热水流速"]

# 获取文件夹下所有txt文件的文件名
txt_files = [f for f in os.listdir(txt_folder_path) if f.endswith('.txt')]

# 随机选择50个txt文件
selected_files = random.sample(txt_files, 100)

# 初始化一个空的DataFrame来存放所有选择的批次数据
all_data = pd.DataFrame()

# 遍历选中的文件并读取数据
for file in selected_files:
    file_path = os.path.join(txt_folder_path, file)

    # 读取txt文件并将其转换为DataFrame，使用正则表达式匹配一个或多个空格
    data = pd.read_csv(file_path, sep='\s+', header=None)

    # 检查当前文件的列数与预定义列名数目是否匹配
    num_columns = data.shape[1]  # 当前txt文件的列数

    # 如果列数不匹配，则打印并进行处理
    if num_columns == len(columns):
        data.columns = columns  # 如果列数匹配，直接使用预定义列名
    else:
        print(f"文件 {file} 的列数为 {num_columns}，不匹配预定义列名数目。")
        # 动态生成列名
        data.columns = [f"Column{i + 1}" for i in range(num_columns)]  # 动态生成列名

    # 可以在这里做一些数据处理，比如为每个批次添加一个标识
    # data['Batch'] = file  # 可选，添加一个批次列

    # 将数据添加到all_data中
    all_data = pd.concat([all_data, data], ignore_index=True)

# 将合并后的数据写入Excel
all_data.to_excel(output_excel_path, index=False)

print(f"数据已成功保存到 {output_excel_path}")
