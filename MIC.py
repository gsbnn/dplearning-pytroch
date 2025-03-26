import pandas as pd
import numpy as np
from minepy import MINE

data_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
select_channels = "B:N,P:Q"
label = "产物浓度"
data = pd.read_excel(data_path, header=0, index_col=None, usecols=select_channels) # 忽略采样序号和热流速度

def get_mic(dataframe, mic_path):
    y_label = dataframe.columns.to_list()
    print(y_label)
    data = np.array(dataframe)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std
    n = len(y_label)
    mine = MINE(alpha=0.6, c=15)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mine.compute_score(data[:, i], data[:, j])
            result[i, j] = mine.mic()
            result[j, i] = result[i, j]

    RT = pd.DataFrame(result, index=y_label, columns=y_label)
    RT.to_excel(mic_path, header=True, index=True)

get_mic(data, 'data\mic_result_1.xlsx')