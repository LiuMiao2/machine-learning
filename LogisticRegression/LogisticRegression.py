"""

@Author: sanshui
@Time:2021/11/15 15:15
@Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def get_data():
    """
    获取数据集
    :return:
    """
    iris = load_iris()

    return iris.data, iris.target


def split_data(data, target):
    """
    划分数据集
    :param data:
    :param target:
    :return:
    """
    print(type(data))
    target = np.reshape(target, (target.shape[0], 1))

    # 正则化数据，防止数据大小本身对结果造成影响
    sd = StandardScaler()
    data = sd.fit_transform(data)

    # 拼接特征值与类别
    dataset = np.hstack((data, target))
    n = dataset.shape[0]

    # 打乱数据
    np.random.shuffle(dataset)

    # 划分数据集，返回训练集与测试集
    train = dataset[:int(0.7 * n), :]
    test = dataset[int(0.7 * n):, :]

    return train, test


def sigmoid(z):
    """
    sigmoid函数
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


def draw_sigmoid():
    """
    画出sigmoid函数
    :return:
    """
    fig, ax = plt.subplots()
    x_data = np.arange(-10, 10, 0.1)
    ax.plot(x_data, sigmoid(x_data))
    plt.show()


def calCost(dataset, theta):
    """
    计算代价函数
    :param dataset:
    :param theta:
    :return:
    """
    x = dataset[:, :-1]
    y = dataset[:, -1:]
    z = x @ theta.T
    # 训练数据个数,或者用m = y.shape[1]
    m = y.size
    para1 = np.multiply(-y, np.log(sigmoid(z)))
    para2 = np.multiply((1 - y), np.log(1 - sigmoid(z)))
    # 代价函数Y
    J = 1 / m * np.sum(para1 - para2)
    return J


def gradient(dataset, theta, iters, alpha):
    """
    梯度下降
    :param dataset:
    :param theta:
    :param iters:
    :param alpha:
    :return:
    """
    # 存放每次梯度下降后的损失值
    x = dataset[:, :-1]
    y = dataset[:, -1:]
    for i in range(iters):
        h_x = sigmoid(x @ theta.T)
        theta = theta - alpha / len(x) * (h_x - y).T @ x
    return theta


def get_per_classify_data(data, i):
    """
    返回第i类的数据
    :param data:数据集
    :param i:类别
    :return:
    """
    return data[data[:, -1] == i]


def get_final_theta(data, i, theta, iters, alpha):
    """
    获取梯度下降后的theta值
    :param data:
    :param i:
    :param theta:
    :param iters:
    :param alpha:
    :return:
    """
    dataset = get_per_classify_data(data, i)
    return gradient(dataset, theta, iters, alpha)


def predict(dataset, theta_list):
    """
    预测结果
    :param dataset:
    :param theta_list:
    :return:
    """
    x = dataset[:, :-1]
    per_theta_list = [i[0] for i in theta_list]
    per_theta_list = np.array(per_theta_list)

    per_prob = sigmoid(np.dot(x, per_theta_list.T))

    # 返回每行最大值所在的索引，即概率最大的类别
    return np.argmax(per_prob, axis=1)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    data, target = get_data()
    train, test = split_data(data, target)
    draw_sigmoid()

    iters = 1000 # 迭代次数
    alpha = 0.5 # 学习率
    theta_list = []
    for i in range(data.shape[1]):
        theta = np.zeros((1, data.shape[1]))
        theta_list.append(theta)

    final_theta_list = []
    target_list = list(set(target))

    for i in target_list:
        theta_i = get_final_theta(train, i, theta_list[target_list.index(i)], iters, alpha)
        final_theta_list.append(theta_i)

    y_predict = predict(test, final_theta_list)

    # 查看预测准确度
    print(classification_report(y_predict, test[:, -1]))
