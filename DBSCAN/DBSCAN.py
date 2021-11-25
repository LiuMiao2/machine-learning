"""

@Author: violet
@Time:2021/11/15 15:16
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/dataset1.csv", header=None)
data = data.values.tolist()

# 画出原始图像
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ax.scatter([i[0] for i in data], [i[1] for i in data])
ax.set_title("原始图像")
plt.show()

'''
计算欧氏距离
'''


def calDist(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.dot((a - b), (a - b).T))
    return dist


'''
获取某个点邻域内的点的个数以及列表
'''


def findNeibors(p, epsilon, data):
    neibors = []  # 用于存储这个点邻域内的点
    for q in data:
        dist = calDist(p, q)
        if dist <= epsilon:
            neibors.append(q)
    cnt = len(neibors)
    return cnt, neibors


'''
返回一个未被选择点.若没有未选择点，返回-1
'''


def getUnvisited(selected):
    for i in range(len(selected)):
        if selected[i] == 0:
            return i

    return -1


'''
判断是否将q添加到簇中
'''


def isInClusters(q, all_clusters):
    for clusters in all_clusters:
        if q in clusters:
            return True

    return False


# 设置参数
minPts = 3
epsilon = 1.0

'''
DBSCAN算法
'''


def DBSCAN(epsilon, minPts, data):
    all_clusters = []  # 所有簇
    noiseList = []
    selected = [0 for i in range(len(data))]
    while getUnvisited(selected) != -1:
        C = []  # 保存同一个簇的点
        i = getUnvisited(selected)  # 找未选择点
        selected[i] = 1  # 修改选择状态
        p = data[i]
        cnts, neibors = findNeibors(p, epsilon, data)  # 获取邻域内的点
        if cnts > minPts:  # p为核心点
            C.append(p)  # 将p添加到簇中
            for q in neibors:  # 遍历核心点p的邻域点
                if selected[data.index(q)] == 0:
                    selected[data.index(q)] = -1
                q_cnt, q_neibors = findNeibors(q, epsilon, data)
                if q_cnt > minPts:  # 如果q是核心点，将其邻域内的点添加到neibors中
                    for i in q_neibors:
                        if i not in neibors:
                            neibors.append(i)
                # 判断q是否已经添加到簇
                if not isInClusters(q, all_clusters):
                    C.append(q)
        else:
            noiseList.append(p)

        if len(C) != 0:
            all_clusters.append(C)  # 找完一个簇，添加到all_clusters中

    all_clusters.append(noiseList)  # 将噪声点添加到all_clusters中

    return all_clusters


if __name__ == "__main__":
    all_clusters = DBSCAN(epsilon, minPts, data)
    fig, ax = plt.subplots()
    n = len(all_clusters)
    for i in range(len(all_clusters)):
        cluster = all_clusters[i]
        if i != len(all_clusters) - 1:
            ax.scatter([j[0] for j in cluster], [j[1] for j in cluster])
        else:
            ax.scatter([j[0] for j in cluster], [j[1] for j in cluster], label="noise", c='purple')
            plt.legend()
    ax.set_title("DBSCAN实现")
    plt.show()

    # 调库实现DBSCAN
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=epsilon, min_samples=minPts)
    db.fit(data)
    fig, ax = plt.subplots()
    ax.scatter([i[0] for i in data], [i[1] for i in data], c=db.labels_)
    ax.set_title("调库实现DBSACN")
    plt.show()
