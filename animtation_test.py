import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# 定义生成数据的函数
def data_gen(max_range):
    # 使用itertools.count()生成无限递增的计数器
    for cnt in itertools.count():
        # 当计数器超过最大范围时停止生成数据
        if cnt > max_range:
            break
        print(cnt)
        # 计算时间t和对应的y值，使用np.sin()计算sin函数，np.exp()计算指数函数
        t = cnt / 10
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)

# 初始化函数，设置坐标轴范围和清空数据
def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,


# 创建图形对象以及子图对象
fig, ax = plt.subplots()
# 创建线条对象
line, = ax.plot([], [], lw=2)
# 创建文本对象用于显示 x 和 y 值
text = ax.text(0., 0., '', transform=ax.transAxes)
# 设置文本位置
text.set_position((0.7, 0.95))
# 将文本对象添加到图形中
ax.add_artist(text)
ax.grid()
xdata, ydata = [], []

# 更新函数，将新的数据添加到图形中
def run(data):
    # 获取传入的数据
    t, y = data
    # 将时间和对应的y值添加到xdata和ydata中
    xdata.append(t)
    ydata.append(y)
    # 获取当前坐标轴的范围
    xmin, xmax = ax.get_xlim()
    # 更新文本对象的值
    text.set_text('x = {:.2f}, y = {:.2f}'.format(t, y))
    # 如果时间t超过当前范围，更新坐标轴范围
    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        # 重绘图形
        ax.figure.canvas.draw()
    # 更新线条的数据
    line.set_data(xdata, ydata)

    return line, text

# 创建动画对象
# fig：图形对象
# run：更新函数，用于更新图形中的数据
# data_gen(20)：生成器函数，产生数据的最大范围为20
# interval=100：每帧动画的时间间隔为100毫秒
# init_func=init：初始化函数，用于设置图形的初始状态
# repeat=True：动画重复播放
ani = animation.FuncAnimation(fig, run, data_gen(20), interval=100, init_func=init, repeat=True)

# 显示图形
plt.show()