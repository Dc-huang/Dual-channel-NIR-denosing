import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件
data = np.loadtxt("loss_pre.txt")

plt.title("pre-loss")
plt.xlabel("X train_num")
plt.ylabel("Y loss")

plt.ylim(0, 10)
# 绘制线图
plt.plot(data)
plt.show()