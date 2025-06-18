import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from skrvm import RVR  # 导入 RVR（Relevance Vector Regression）


def create_sliding_windows(data, labels, window_size, step_size):
    num_samples = (len(data) - window_size) // step_size
    windows = np.array([data[i:i + window_size] for i in range(0, num_samples * step_size, step_size)])
    labels = np.array([labels[i + window_size] for i in range(0, num_samples * step_size, step_size)])
    return windows, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Window_size", type=int, default=8)
    args = parser.parse_args()

    # 数据加载
    phase_time = 115
    data = pd.read_excel("dataset/Selected_Columns/pensimdata_phase4_50_batch.xlsx")

    # 提取特征数据
    data_feature = data.iloc[:, :-1].values

    train_features = data_feature[phase_time * 0:phase_time * 20, :]
    test_features = data_feature[phase_time * 20:phase_time * 22, :]
    eval_features = data_feature[phase_time * 23:phase_time * 24, :]

    # 数据标准化
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)
    train_std[train_std == 0] = 1e-2
    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std
    eval_features = (eval_features - train_mean) / train_std
    # # 给训练数据增加噪声
    # noise_factor = 0.2  # 调整噪声强度，通常值在0.01到0.1之间
    # noise = np.random.normal(0, noise_factor, train_features.shape)
    # train_features += noise
    # 标签数据处理
    data_label = data.iloc[:, -1].values.reshape(-1, 1)
    train_labels = data_label[phase_time * 0:phase_time * 20, -1]
    test_labels = data_label[phase_time * 20:phase_time * 22, -1]
    eval_labels = data_label[phase_time * 23:phase_time * 24, -1]

    # 创建滑动窗口
    train_windows, train_window_labels = create_sliding_windows(train_features, train_labels,
                                                                window_size=args.Window_size, step_size=1)
    test_windows, test_window_labels = create_sliding_windows(test_features, test_labels, window_size=args.Window_size,
                                                              step_size=1)

    # 使用 RVR 模型
    rvr_model = RVR(kernel='rbf', n_iter=3000)

    # 训练 RVR 模型
    rvr_model.fit(train_windows.reshape(train_windows.shape[0], -1), train_window_labels)

    # 测试模型
    predictions = rvr_model.predict(test_windows.reshape(test_windows.shape[0], -1))
    test_r2 = r2_score(test_window_labels, predictions)
    test_mae = mean_absolute_error(test_window_labels, predictions)
    test_mse = mean_squared_error(test_window_labels, predictions)
    test_rmse = np.sqrt(np.mean((test_window_labels - predictions) ** 2))

    print(f"R2: {test_r2}   MAE: {test_mae}   MSE: {test_mse}   RMSE: {test_rmse}")

    # 绘制预测结果与实际值的对比
    plt.figure()
    plt.plot(predictions, label="Predictions")
    plt.plot(test_window_labels, label="Actual")
    plt.legend()
    plt.show()

    # 评估模型在评估集上的性能
    eval_windows, eval_labels = create_sliding_windows(eval_features, eval_labels, window_size=8, step_size=1)
    eval_predictions = rvr_model.predict(eval_windows.reshape(eval_windows.shape[0], -1))
    test_r2 = r2_score(eval_labels, eval_predictions)
    test_mae = mean_absolute_error(eval_labels,eval_predictions)
    test_mse = mean_squared_error(eval_labels, eval_predictions)
    test_rmse = np.sqrt(np.mean((eval_labels - eval_predictions) ** 2))

    print(f"R2: {test_r2}   MAE: {test_mae}   MSE: {test_mse}   RMSE: {test_rmse}")
    # 绘制评估集的预测结果
    plt.figure()
    plt.title("Evaluation")
    plt.plot(eval_predictions, label="Predictions")
    plt.plot(eval_labels, label="Actual")
    plt.legend()
    plt.show()
